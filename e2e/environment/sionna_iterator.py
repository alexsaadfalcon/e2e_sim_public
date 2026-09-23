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
# Legacy artifact (traced at Sionna's 3.5 GHz default -- see notes/ESTABLISHED_FACTS.md
# F93). Never modified/deleted by this module; kept selectable for comparison.
SIONNA_MUNICH_LEGACY_PATH = os.path.join(_this_dir, 'sionna_sims', 'munich.pkl')
# Ka-band re-trace (28.5-31.5 GHz around a 30 GHz carrier, the owner's band decision
# 2026-09-23) written by `e2e.environment.sionna_simple_channel`. This is what plain
# 'munich' now resolves to, when present.
SIONNA_MUNICH_KA_PATH = os.path.join(_this_dir, 'sionna_sims', 'munich_ka.pkl')
# Back-compat alias some call sites may still reference.
SIONNA_MUNICH_PATH = SIONNA_MUNICH_LEGACY_PATH
# Special `link` value that selects the legacy 3.5 GHz file through the same
# `SionnaMunichIterator(link=...)`/`SionnaEnvironmentBlock('munich', link=...)` call site
# -- lets a caller pick the legacy artifact without a second scenario name.
MUNICH_LEGACY_LINK = 'munich_legacy_3p5ghz'


# Factories forward an optional `link` selector to SionnaIterator so a multi-link pkl can
# be addressed explicitly. They take no positional args, so existing `SionnaMunichIterator()`
# / `SionnaEtoileIterator()` call sites keep working (link=None -> first link / single array).
def SionnaEtoileIterator(link=None):
    return SionnaIterator(SIONNA_ETOILE_PATH, link=link)


def SionnaMunichIterator(link=None):
    # `link=MUNICH_LEGACY_LINK` is a FILE selector here, not a pkl-internal link name
    # (the legacy pkl is a bare ndarray with no links at all) -- it exists so the legacy
    # 3.5 GHz artifact stays reachable through 'munich' without a second scenario name.
    if link == MUNICH_LEGACY_LINK:
        return SionnaIterator(SIONNA_MUNICH_LEGACY_PATH, link=None)
    if os.path.exists(SIONNA_MUNICH_KA_PATH):
        return SionnaIterator(SIONNA_MUNICH_KA_PATH, link=link)
    return SionnaIterator(SIONNA_MUNICH_LEGACY_PATH, link=link)

