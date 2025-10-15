import torch

# numerical tolerance
tol = 1e-4

def subspace_dist(A, B):
    # PyTorch doesn't have a direct equivalent to scipy.linalg.subspace_angles
    # We'll implement it using SVD
    assert A.shape == B.shape, f'shape mismatch: {A.shape} and {B.shape}'
    AtB = A.t().conj() @ B
    S = torch.linalg.svd(AtB, full_matrices=False).S
    S = torch.clamp(S, -1, 1)  # Ensure numerical stability
    thetas = torch.arccos(S)
    return torch.sqrt(torch.sum(thetas ** 2))

def subspace_dist_frob(A, B):
    assert A.shape == B.shape, f'shape mismatch: {A.shape} and {B.shape}'
    d = A.shape[1]
    ip = torch.linalg.norm(A.t().conj() @ B) ** 2
    ip = min(d, ip.item())
    return torch.sqrt(torch.tensor(d - ip))

def Glog(X, Y):
    # first argument is the base
    # H = Glog(X, Y) <=> Y = Gexp(X, H)
    assert torch.linalg.norm(X.t().conj() @ X - torch.eye(X.shape[1], device=X.device)) < tol
    assert torch.linalg.norm(Y.t().conj() @ Y - torch.eye(Y.shape[1], device=Y.device)) < tol
    n, p = X.shape
    XH = X.t().conj()
    X_null = torch.eye(n, device=X.device) - X @ XH
    XY_inv = torch.linalg.inv(XH @ Y)
    U, E, V = torch.linalg.svd((X_null @ Y) @ XY_inv, full_matrices=False)
    Theta = torch.diag(torch.atan(E))
    Gl = U @ Theta @ V.t().conj()
    return Gl

def Gexp(X, H):
    if torch.any(torch.isnan(H)) or torch.any(torch.isnan(X)):
        raise ValueError("H or X contains NaN values")
    assert torch.linalg.norm(X.t().conj() @ H) < tol, f"X.t().conj() @ H = {torch.linalg.norm(X.t().conj() @ H)}"
    assert torch.linalg.norm(X.t().conj() @ X - torch.eye(X.shape[1], device=X.device)) < tol
    U, E, VH = torch.linalg.svd(H, full_matrices=False)
    V = VH.t().conj()
    C = torch.diag(torch.cos(E)).cfloat()
    S = torch.diag(torch.sin(E)).cfloat()
    Ge = ((X @ V) @ C + U @ S) @ VH
    return Ge

def parallel_transport(X, H, Delta):
    U, E, VH = torch.linalg.svd(H, full_matrices=False)
    V = VH.t().conj()
    UH = U.t().conj()
    C = torch.diag(torch.cos(E)).cfloat()
    S = torch.diag(torch.sin(E)).cfloat()
    tau_Delta = (-X @ V @ S @ UH + U @ C @ UH) @ Delta
    tau_Delta += Delta - U @ (UH @ Delta)
    return tau_Delta

if __name__ == '__main__':
    pass
