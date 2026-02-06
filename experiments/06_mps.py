import torch

# -------- Device setup --------
def set_device(device="cpu"):
    return torch.device(device)

# -------- Utility: represent complex numbers as real pairs --------
def complex_mm(a_real, a_imag, b_real, b_imag):
    """Matrix multiply (a+ib)(b+ic) = (ab - ic) + i(ad + bc)."""
    real = a_real @ b_real - a_imag @ b_imag
    imag = a_real @ b_imag + a_imag @ b_real
    return real, imag

# -------- Basic gates (real and imaginary parts) --------
def get_gates_real(device):
    sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=torch.float32, device=device))
    H_real = torch.tensor([[1, 1], [1, -1]], dtype=torch.float32, device=device) / sqrt2
    H_imag = torch.zeros_like(H_real)
    I_real = torch.eye(2, dtype=torch.float32, device=device)
    I_imag = torch.zeros_like(I_real)
    CNOT_real = torch.tensor([[1,0,0,0],
                              [0,1,0,0],
                              [0,0,0,1],
                              [0,0,1,0]], dtype=torch.float32, device=device)
    CNOT_imag = torch.zeros_like(CNOT_real)
    return (I_real, I_imag), (H_real, H_imag), (CNOT_real, CNOT_imag)

# -------- MPS initialization --------
def init_product_mps_real(state_vecs, device):
    mps_real, mps_imag = [], []
    for sv in state_vecs:
        sv = torch.tensor(sv, dtype=torch.float32, device=device)
        A_real = sv.view(1, 2, 1)
        A_imag = torch.zeros_like(A_real)
        mps_real.append(A_real)
        mps_imag.append(A_imag)
    return mps_real, mps_imag

# -------- Convert MPS to full state (real/imag parts) --------
def mps_to_state_real(mps_real, mps_imag):
    psi_r = mps_real[0].reshape(2, -1)
    psi_i = mps_imag[0].reshape(2, -1)
    for A_r, A_i in zip(mps_real[1:], mps_imag[1:]):
        chi_l, d, chi_r = A_r.shape
        psi_r = psi_r.reshape(-1, chi_l)
        psi_i = psi_i.reshape(-1, chi_l)
        # complex tensor contraction
        r1, i1 = complex_mm(psi_r, psi_i, A_r.reshape(chi_l, d*chi_r), A_i.reshape(chi_l, d*chi_r))
        psi_r, psi_i = r1.reshape(-1, chi_r), i1.reshape(-1, chi_r)
    return psi_r.reshape(-1), psi_i.reshape(-1)

# -------- Apply single-qubit gate --------
def apply_one_site_real(mps_real, mps_imag, G_real, G_imag, site):
    A_r, A_i = mps_real[site], mps_imag[site]
    chiL, d, chiR = A_r.shape
    # (p',a,b) = sum_q G_{p'q} A_{a,q,b}
    Gr = G_real
    Gi = G_imag
    Ar = A_r.permute(1,0,2).reshape(d, -1)
    Ai = A_i.permute(1,0,2).reshape(d, -1)
    new_r, new_i = complex_mm(Gr, Gi, Ar, Ai)
    new_r = new_r.reshape(2, chiL, chiR).permute(1,0,2)
    new_i = new_i.reshape(2, chiL, chiR).permute(1,0,2)
    mps_real[site], mps_imag[site] = new_r.contiguous(), new_i.contiguous()
    return mps_real, mps_imag

# -------- Fixed two-site gate (real dtype) --------
def apply_two_site_real(mps_real, mps_imag, U_real, U_imag, site, chi_max=None, eps=1e-6):
    A_r, A_i = mps_real[site], mps_imag[site]
    B_r, B_i = mps_real[site+1], mps_imag[site+1]
    chiL, d1, chiM = A_r.shape
    chiM2, d2, chiR = B_r.shape
    assert chiM == chiM2
    # combine tensors
    Theta_r = torch.tensordot(A_r, B_r, dims=([2],[0])) - torch.tensordot(A_i, B_i, dims=([2],[0]))
    Theta_i = torch.tensordot(A_r, B_i, dims=([2],[0])) + torch.tensordot(A_i, B_r, dims=([2],[0]))
    Theta_r = Theta_r.reshape(chiL, d1*d2, chiR)
    Theta_i = Theta_i.reshape(chiL, d1*d2, chiR)
    # apply U
    Ur, Ui = U_real.view(d1*d2, d1*d2), U_imag.view(d1*d2, d1*d2)
    r1, i1 = complex_mm(Theta_r.reshape(-1, d1*d2), Theta_i.reshape(-1, d1*d2), Ur, Ui)
    Theta_r = r1.reshape(chiL*d1, d2*chiR)
    Theta_i = i1.reshape(chiL*d1, d2*chiR)
    # compute magnitude matrix for SVD
    Theta_mag = torch.sqrt(Theta_r**2 + Theta_i**2)
    U_s, S, Vh = torch.linalg.svd(Theta_mag, full_matrices=False)
    if chi_max is not None:
        keep = min(int((S > eps).sum().item()), chi_max)
    else:
        keep = int((S > eps).sum().item())
    U_s = U_s[:, :keep]
    S = S[:keep]
    Vh = Vh[:keep, :]
    # rebuild approximate real and imag parts
    A_new_r = U_s.reshape(chiL, d1, keep)
    A_new_i = torch.zeros_like(A_new_r)
    B_new_r = (S.view(-1,1) * Vh).reshape(keep, d2, chiR)
    B_new_i = torch.zeros_like(B_new_r)
    mps_real[site], mps_imag[site] = A_new_r, A_new_i
    mps_real[site+1], mps_imag[site+1] = B_new_r, B_new_i
    return mps_real, mps_imag

# -------- Normalize --------
def normalize_mps_real(mps_real, mps_imag):
    psi_r, psi_i = mps_to_state_real(mps_real, mps_imag)
    norm_val = torch.sum(psi_r**2 + psi_i**2)
    scale = 1.0 / torch.sqrt(norm_val)
    mps_real[0] *= scale
    mps_imag[0] *= scale
    return mps_real, mps_imag

# -------- Example --------
device = set_device("cpu")  # change to your device
(Ir, Ii), (Hr, Hi), (CNOTr, CNOTi) = get_gates_real(device)
sv0 = [1.0, 0.0]
sv1 = [1.0, 0.0]
mps_real, mps_imag = init_product_mps_real([sv0, sv1], device)
mps_real, mps_imag = apply_one_site_real(mps_real, mps_imag, Hr, Hi, 0)
mps_real, mps_imag = apply_two_site_real(mps_real, mps_imag, CNOTr, CNOTi, 0, chi_max=4)
mps_real, mps_imag = normalize_mps_real(mps_real, mps_imag)

psi_r, psi_i = mps_to_state_real(mps_real, mps_imag)
print("Real part:", psi_r)
print("Imag part:", psi_i)

