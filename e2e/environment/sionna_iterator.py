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
        # Three payload shapes on disk (see module docstring / CLAUDE.md format contract):
        #   1. bare ndarray                          -> legacy single-link
        #   2. dict WITHOUT "meta"  {name: array}     -> legacy multi-link
        #   3. dict WITH "meta" AND "links"           -> v2 self-describing multi-link
        if isinstance(data, dict) and "meta" in data and "links" in data:
            self.meta = data["meta"]
            links_dict = data["links"]
            self.links = list(links_dict.keys())
            if link is None:
                link = self.links[0]
            elif link not in links_dict:
                raise KeyError(f"link {link!r} not in {fname} (available: {self.links})")
            self.link = link
            self.all_s_pars = links_dict[link]
            self.link_meta = self.meta.get("links", {}).get(link)
        elif isinstance(data, dict):
            self.meta = None
            self.link_meta = None
            self.links = list(data.keys())
            if link is None:
                link = self.links[0]
            elif link not in data:
                raise KeyError(f"link {link!r} not in {fname} (available: {self.links})")
            self.link = link
            self.all_s_pars = data[link]
        else:
            self.meta = None
            self.link_meta = None
            self.links = None
            self.link = None
            self.all_s_pars = data

    @property
    def freq_plan(self):
        """The frequency plan dict ({carrier_hz,start_hz,stop_hz,num_freqs}) for v2
        payloads, or None for legacy pkls / when unavailable."""
        if self.meta is None:
            return None
        return self.meta.get("freq_plan")

    @property
    def rx_array_shape(self):
        """The selected link's rx array shape (rows, cols) for v2 payloads, else None."""
        if self.link_meta is None:
            return None
        shape = self.link_meta.get("rx_array_shape")
        return tuple(shape) if shape is not None else None

    @property
    def physical_scale(self):
        """Whether the selected link's frames are physically scaled, for v2 payloads,
        else None."""
        if self.link_meta is None:
            return None
        return self.link_meta.get("physical_scale")

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
        if not isinstance(data, dict):
            return None
        if "meta" in data and "links" in data:
            return list(data["links"].keys())
        return list(data.keys())


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

