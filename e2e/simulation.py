import torch
from tqdm import tqdm
from collections import defaultdict


def get_U_true(s_pars, d):
    assert len(s_pars.shape) == 4
    s_pars_0 = s_pars[:, :, 0, :]
    s_pars_0 = s_pars_0.view(-1, s_pars_0.shape[-1])
    U, _, __ = torch.linalg.svd(s_pars_0)
    U_true = U[:, :d]
    return U_true

def perturb_basis(U):
    U = U + 1e-3 * (torch.randn_like(U) + 1j * torch.randn_like(U))
    return torch.linalg.qr(U)[0]

class Simulation:
    def __init__(self, 
        environment_block,
        downstream_blocks,
        d,
        circuit_block=None,
        interconnect_block=None,
        afe_block=None,
        subspace_block=None,
    ):
        self.environment_block = environment_block
        self.downstream_blocks = downstream_blocks
        self.d = d
        self.circuit_block = circuit_block
        self.interconnect_block = interconnect_block
        self.afe_block = afe_block
        self.subspace_block = subspace_block
        if subspace_block is None and afe_block is not None:
            raise ValueError('Need subspace block to pair with AFE block')
        self.outputs = defaultdict(list)

    def step(self):
        self.environment_block.step()

    def reset(self):
        self.environment_block.reset()

    def feed_forward(self):
        s_pars = self.environment_block.get_S_pars()
        s_pars_orig = s_pars.view(32, 32, 1, -1).clone()
        U_true = get_U_true(s_pars, self.d)
        # short circuit Oja learning to avoid hyperparameter tuning
        self.subspace_block.oja.U = perturb_basis(U_true)

        if self.circuit_block:
            s_pars = torch.cat([s_pars, s_pars], dim=2)
            s_pars, PRX = self.circuit_block.apply_circuit(s_pars)
            s_pars = s_pars[:, :, :1, :]

        assert s_pars.shape[1] == 1, 'MIMO not supported yet'
        assert s_pars.shape[1] == 1, 'Multiple chirps not supported yet'
        s_pars = s_pars.view(32, 32, 1, -1)

        if self.interconnect_block:
            s_pars = self.interconnect_block.apply_interconnect(s_pars)

        if self.afe_block and self.subspace_block:
            V = s_pars.view(-1, s_pars.shape[-1])
            A = self.subspace_block.gen_A_ada()
            Aq, X = self.afe_block.apply_mat_mul(A, V)
            self.subspace_block.update(X, Aq)
            Xt = self.afe_block.reconstruct(Aq, X)
            s_pars = Xt.view(s_pars.shape)
        elif self.subspace_block:
            self.subspace_block.update(s_pars)
        
        state_dict = {
            'U': self.subspace_block.oja.U,
            'U_true': U_true,
            's_pars': s_pars,
            'PRX': PRX,
        }

        for downstream_block in self.downstream_blocks:
            outputs = downstream_block.apply(state_dict)
            for output_name, output in outputs.items():
                self.outputs[output_name].append(output)
        
    def get_outputs(self):
        return self.outputs

    def run(self, n_steps=10):
        self.reset()
        for i in tqdm(range(n_steps), desc='RUNNING ARRAY SIMULATION'):
            self.feed_forward()
            self.step()
        return self.get_outputs()

