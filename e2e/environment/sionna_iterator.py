import os
import pickle
import numpy as np


class SionnaIterator:
    def __init__(self, fname, link=None):
        # Use a context manager so the file handle is closed promptly. Leaving it open
        # (the old `pickle.load(open(...))`) keeps a lock on the .pkl on Windows, which
        # makes regenerating/overwriting the same path later fail with a sharing error.
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        # Multi-link scenarios (e.g. ISAC) dump a dict {link_name: frames_array};
        # single-link scenarios dump a plain ndarray. Select one link's frames so the
        # rest of the pipeline sees the same single-link array either way.
        if isinstance(data, dict):
            self.links = list(data.keys())
            if link is None:
                link = self.links[0]
            elif link not in data:
                raise KeyError(f"link {link!r} not in {fname} (available: {self.links})")
            self.link = link
            self.all_s_pars = data[link]
        else:
            self.links = None
            self.link = None
            self.all_s_pars = data

    def __iter__(self):
        for i in range(self.all_s_pars.shape[0]):
            yield self.all_s_pars[i]

    def __len__(self):
        return self.all_s_pars.shape[0]

    def __getitem__(self, i):
        return self.all_s_pars[i]

    @staticmethod
    def available_links(fname):
        """Return the list of link names in ``fname`` (multi-link pkl), or None for a
        single-array pkl. Lets callers discover selectable links without iterating."""
        # Context manager: close the handle so we don't lock the .pkl (see __init__).
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        return list(data.keys()) if isinstance(data, dict) else None


_this_dir = os.path.abspath(os.path.dirname(__file__))
SIONNA_ETOILE_PATH = os.path.join(_this_dir, 'sionna_sims', 'etoile.pkl')
SIONNA_MUNICH_PATH = os.path.join(_this_dir, 'sionna_sims', 'munich.pkl')


# Factories forward an optional `link` selector to SionnaIterator so a multi-link pkl can
# be addressed explicitly. They take no positional args, so existing `SionnaMunichIterator()`
# / `SionnaEtoileIterator()` call sites keep working (link=None -> first link / single array).
def SionnaEtoileIterator(link=None):
    return SionnaIterator(SIONNA_ETOILE_PATH, link=link)


def SionnaMunichIterator(link=None):
    return SionnaIterator(SIONNA_MUNICH_PATH, link=link)

