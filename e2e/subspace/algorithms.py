import torch
import torch.linalg as linalg
from tqdm import tqdm
import matplotlib.pyplot as plt

from .subspace_utils import Gexp, subspace_dist_frob


norm = linalg.norm
orth = lambda A: linalg.qr(A)[0]
n_iter = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def randn_complex(a, b, device=device):
    # Measured E|X|^2 = 2.0 over 1e6 samples (real/imag parts are each unit
    # variance, i.e. CN(0, 2), not CN(0, 4) as a stacked-cfloat construction
    # might suggest: torch.randn(..., dtype=torch.cfloat) itself already
    # splits unit variance across its real/imag parts). This real-dtype form
    # is the simplest, self-documenting way to reproduce that same E|X|^2 = 2
    # scale, so seeded downstream behavior is unchanged.
    return torch.randn(a, b, device=device) + 1j * torch.randn(a, b, device=device)

def rand_orth_complex(d, k, device=device):
    U = randn_complex(d, k, device=device)
    return orth(U)


class GROUSE(object):
    def __init__(self, d, k, eta=None, fixed_step=False, device=device):
        self.d = d
        self.k = k
        self.U = rand_orth_complex(d, k, device=device)

        if eta is not None:
            self.eta0 = eta
        else:
            self.eta0 = 10.0
        self.fixed_step = fixed_step
        self.it = 1.0

    def get_step(self, X, A):
        U = self.U

        AU = A @ U

        w = torch.linalg.pinv(AU) @ X
        p = U @ w
        r = X - A @ p
        r = A.T.conj() @ r

        sigma = norm(r) * norm(p)
        if self.fixed_step:
            eta = self.eta0
        else:
            eta = self.eta0 / self.it

        pnorm = norm(p)
        rnorm = norm(r)
        wnorm = norm(w)
        theta = eta * sigma

        pw = pnorm * wnorm
        rw = rnorm * wnorm

        if pw == 0 or rw == 0: raise ValueError

        step = (torch.sin(theta) * r / rnorm + (torch.cos(theta) - 1) * p / pnorm) @ (w.T.conj() / wnorm)

        return step, w

    def add_data(self, X, A):
        for _ in range(n_iter):
            step, w = self.get_step(X, A)
            self.U += step
            self.U = orth(self.U)
        self.Y = self.U
        self.it += 1.0
        return self.U

class Oja(object):
    def __init__(self, d, k, eta=None, fixed_step=False, device=device):
        self.d = d
        self.k = k
        self.U = rand_orth_complex(d, k, device=device)

        if eta is not None:
            self.eta0 = eta
        else:
            self.eta0 = 10.0
        self.fixed_step = fixed_step
        self.it = 1.0

    def get_grad(self, X, A):
        U = self.U

        AU = A @ U

        w = torch.linalg.pinv(AU) @ X
        p = U @ w
        r = X - A @ p
        r = A.T.conj() @ r
        x_tilde = p + r

        grad = -x_tilde @ w.T.conj()

        return grad, w

    def add_data(self, X, A):
        if self.fixed_step:
            eta = self.eta0
        else:
            eta = self.eta0 / self.it

        grad, w = self.get_grad(X, A)
        grad /= torch.norm(grad)
        step = -eta * grad
        self.U += step
        self.U = orth(self.U)
        self.it += 1.0
        return self.U

    def compute_loss(self, grad, eta, X, A):
        step = -eta * grad
        U = self.U + step
        U = orth(U)
        AU = A @ U
        W = torch.linalg.pinv(AU) @ X
        return torch.norm(X - AU @ W) ** 2

    def add_data_linesearch(self, X, A):
        grad, w = self.get_grad(X, A)
        grad /= torch.norm(grad)
        etas = torch.logspace(-3, 1, 100)
        losses = []
        for eta in etas:
            loss = self.compute_loss(grad, eta, X, A)
            losses.append(loss.cpu().item())

        # plt.figure()
        # plt.plot(etas, losses)
        # plt.semilogx()
        # plt.show()
        eta_min = etas[torch.argmin(torch.tensor(losses))]
        step = -eta_min * grad
        self.U += step
        self.U = orth(self.U)
        self.it += 1.0
        return self.U

    def add_data_ternarysearch(self, X, A):
        grad, w = self.get_grad(X, A)
        grad /= torch.norm(grad)
        
        eta_min, eta_max = 1e-4, 1e2
        loss_min = self.compute_loss(grad, eta_min, X, A)
        loss_max = self.compute_loss(grad, eta_max, X, A)
        max_iterations = 30
        tolerance = 1e-6

        for _ in range(max_iterations):
            eta_left = eta_min + (eta_max - eta_min) / 3
            eta_right = eta_max - (eta_max - eta_min) / 3
            
            loss_left = self.compute_loss(grad, eta_left, X, A)
            loss_right = self.compute_loss(grad, eta_right, X, A)
            # print('etas', eta_min, eta_left, eta_right, eta_max)
            # print('losses', loss_min, loss_left, loss_right, loss_max)
            # input()

            if loss_left < loss_right:
                eta_max = eta_right
                loss_max = loss_right
            else:
                eta_min = eta_left
                loss_min = loss_left

            if abs(loss_max - loss_min) < tolerance:
                eta_optimal = (eta_min + eta_max) / 2
                break
        else:
            # If we've exhausted all iterations, choose the midpoint
            eta_optimal = (eta_min + eta_max) / 2

        step = -eta_optimal * grad
        self.U += step
        self.U = orth(self.U)
        self.it += 1.0
        return self.U


class GrassmannGD(object):
    def __init__(self, d, k, eta=None, fixed_step=False, device=device):
        self.d = d
        self.k = k
        self.U = rand_orth_complex(d, k, device=device)

        if eta is not None:
            self.eta0 = eta
        else:
            self.eta0 = 10.0
        self.fixed_step = fixed_step
        self.it = 1.0

    def add_data(self, X, A):
        self.U = orth(self.U)
        for _ in range(n_iter):
            U = self.U

            AU = A @ U
            W = torch.linalg.pinv(AU) @ X
            R = A @ U @ W - X
            RT = A.T.conj() @ R
            if self.fixed_step:
                eta = self.eta0
            else:
                eta = self.eta0 / self.it
            grad = RT @ W.T.conj()
            grad /= torch.norm(grad)
            step = -eta * grad
            step = step - self.U @ (self.U.T.conj() @ step)
            self.U = Gexp(self.U, step)
            self.U = orth(self.U)

        self.U = orth(self.U)
        self.it += 1.0
        return self.U


def gen_A_ada(U, m, device=None):
    # Build the sensing matrix on U's own device (not the module-level default) so a
    # CPU-resident U on a CUDA box doesn't device-mismatch in the matmul below.
    if device is None:
        device = U.device
    d, k = U.shape
    B = torch.randn(d, m - k, dtype=torch.cfloat, device=device)
    B -= U @ (U.T.conj() @ B)
    # U's columns are unit-norm (orthonormal), so the deterministic rows of A
    # (U^H) already have unit row-norm. The random columns of B are unnormalized
    # complex Gaussians, whose column norm scales like sqrt(n) (measured ~32 for
    # n=1024) -- a huge scale mismatch against the deterministic rows if left
    # as-is. Normalize so both halves of the sensing matrix weight equally.
    B = B / B.norm(dim=0, keepdim=True)
    A = torch.zeros((m, d), dtype=torch.cfloat, device=device)
    A[:k, :] = U.T.conj()
    A[k:, :] = B.T.conj()
    return A

if __name__ == '__main__':
    # test GROUSE, Oja, and GrassmannGD
    n = 100
    m = 20
    d = 10
    b = 10
    sigma = 1e-2
    eta = 0.1
    nsteps = 1000
    fixed_step = False
    U = rand_orth_complex(n, d)

    grouse = GROUSE(n, d, eta, fixed_step)
    oja = Oja(n, d, eta, fixed_step)
    ggd = GrassmannGD(n, d, eta, fixed_step)
    ada_grouse = GROUSE(n, d, eta, fixed_step)
    ada_oja = Oja(n, d, eta, fixed_step)
    ada_ggd = GrassmannGD(n, d, eta, fixed_step)

    algos = dict(grouse=grouse, oja=oja, ggd=ggd, ada_grouse=ada_grouse, ada_oja=ada_oja, ada_ggd=ada_ggd)
    algo_Ys = {algo_name: [] for algo_name in algos}

    # test non-adaptive
    for t in tqdm(range(nsteps)):
        coeffs = randn_complex(d, b)
        noise = randn_complex(n, b) * sigma
        V = U @ coeffs + noise
        A = randn_complex(m, n)
        X = A @ V
        for algo_name, algo in algos.items():
            if 'ada' in algo_name:
                ada_A = gen_A_ada(algo.U.clone(), m)
                ada_X = ada_A @ V
                algo.add_data(ada_X, ada_A)
            else:
                algo.add_data(X, A)
            algo_Ys[algo_name].append(algo.U.clone())

    # compute non-adaptive errors
    algo_errors = {algo_name: [] for algo_name in algos}
    for algo_name, algo_Ys in algo_Ys.items():
        for Y in algo_Ys:
            err = subspace_dist_frob(Y, U) ** 2
            algo_errors[algo_name].append(err)

    # plot non-adaptive errors
    for algo_name, errors in algo_errors.items():
        plt.plot(errors, label=algo_name)
    plt.legend()
    plt.semilogy()
    plt.show()
