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
        array_shape=None,
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
        # Receive-array geometry (n_rx_x, n_rx_y). Explicit arg wins; otherwise take it
        # from the environment block if it advertises one; otherwise default to 32x32
        # (the historical hardcoded size) for backward compatibility.
        if array_shape is None:
            array_shape = getattr(environment_block, 'array_shape', (32, 32))
        self.n_rx_x, self.n_rx_y = array_shape
        self.outputs = defaultdict(list)
        # The online subspace tracker is initialized once (from the first frame's
        # subspace) and then tracks the evolving scene; this flag guards that
        # one-time warm start. See feed_forward.
        self._subspace_started = False

    def step(self):
        self.environment_block.step()

    def reset(self):
        self.environment_block.reset()
        # Re-arm the one-time tracker warm start so each run() starts fresh.
        self._subspace_started = False
        # Downstream blocks with per-run state (e.g. ModemBlock's frame-indexed
        # noise counter) expose reset(); rewind them so repeated run() calls on
        # the same Simulation are reproducible.
        for block in self.downstream_blocks:
            if hasattr(block, "reset"):
                block.reset()

    def feed_forward(self):
        s_pars = self.environment_block.get_S_pars()
        U_true = get_U_true(s_pars, self.d)
        # Initialize the online (Oja) tracker ONCE, from the first frame's subspace
        # (lightly perturbed), then let it actually track the evolving scene across
        # frames. Previously the basis was reset to the ground truth on EVERY frame,
        # which made online tracking a no-op (subspace_err reflected the injected
        # perturbation, not the tracker). Note: because each frame is a fresh
        # moving-platform channel snapshot, the per-frame subspace can change faster
        # than a one-step tracker follows, so subspace_err reflects that tracking lag.
        if self.subspace_block is not None and not self._subspace_started:
            self.subspace_block.oja.U = perturb_basis(U_true)
            self._subspace_started = True

        PRX = None
        if self.circuit_block:
            s_pars = torch.cat([s_pars, s_pars], dim=2)
            s_pars, PRX = self.circuit_block.apply_circuit(s_pars)
            s_pars = s_pars[:, :, :1, :]

        assert s_pars.shape[1] == 1, 'MIMO not supported yet'
        assert s_pars.shape[2] == 1, 'Multiple chirps not supported yet'
        s_pars = s_pars.view(self.n_rx_x, self.n_rx_y, 1, -1)

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
            # No AFE: feed the subspace tracker the full-precision compressed
            # measurements directly (same A-generation as the AFE branch, but
            # without quantization).
            V = s_pars.view(-1, s_pars.shape[-1])
            A = self.subspace_block.gen_A_ada()
            X = A @ V
            self.subspace_block.update(X, A)
        
        reserved_keys = {'U', 'U_true', 's_pars', 'PRX'}
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
            # Make a block's outputs visible to subsequent downstream blocks, so they
            # can compose (e.g. a comms BERBlock consumes a ModemBlock's tx/rx bits).
            # Existing product blocks emit disjoint keys, so this is a no-op for them.
            # Guard the reserved pipeline keys: a block must not clobber the canonical
            # state the orchestrator feeds every block, so the contract is explicit.
            for k in outputs:
                if k in reserved_keys:
                    raise ValueError(
                        f"downstream block {downstream_block} emitted reserved key {k!r}"
                    )
            state_dict.update(outputs)
        
    def get_outputs(self):
        return self.outputs

    def run(self, n_steps=10):
        self.reset()
        for i in tqdm(range(n_steps), desc='RUNNING ARRAY SIMULATION'):
            self.feed_forward()
            self.step()
        return self.get_outputs()

