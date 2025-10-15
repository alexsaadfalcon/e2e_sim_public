import os
import pickle
import numpy as np


class SionnaIterator:
    def __init__(self, fname):
        self.all_s_pars = pickle.load(open(fname, 'rb'))

    def __iter__(self):
        for i in range(self.all_s_pars.shape[0]):
            yield self.all_s_pars[i]

    def __len__(self):
        return self.all_s_pars.shape[0]

    def __getitem__(self, i):
        return self.all_s_pars[i]

_this_dir = os.path.abspath(os.path.dirname(__file__))
SIONNA_ETOILE_PATH = os.path.join(_this_dir, 'sionna_sims', 'etoile.pkl')
SionnaEtoileIterator = lambda: SionnaIterator(SIONNA_ETOILE_PATH)
SIONNA_MUNICH_PATH = os.path.join(_this_dir, 'sionna_sims', 'munich.pkl')
SionnaMunichIterator = lambda: SionnaIterator(SIONNA_MUNICH_PATH)

