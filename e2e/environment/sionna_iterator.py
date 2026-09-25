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
# Ka re-trace in which the LINE OF SIGHT SWEEPS: the array pans frame by frame so the
# arrival azimuth walks -28.40 -> +28.89 deg over 30 frames while the path set, and so
# the frame's rank, stay put (owner directive 2026-09-24; generated and measured in
# notes/LOSWEEP_REPORT_2026-09-25.md). Selected explicitly through MUNICH_LOSWEEP_LINK;
# plain 'munich' NEVER resolves to it, so every other thrust keeps the static file.
SIONNA_MUNICH_LOSWEEP_PATH = os.path.join(_this_dir, 'sionna_sims',
                                          'munich_ka_losweep.pkl')


def _resolve_munich_default_path(ka_path=None, legacy_path=None):
    """Ka-preferred, legacy-fallback resolution, pulled out as a pure function so tests
    can exercise the logic directly (see the docstring on SIONNA_MUNICH_PATH below for
    why the module attribute itself is resolved once, not re-checked per call)."""
    ka_path = SIONNA_MUNICH_KA_PATH if ka_path is None else ka_path
    legacy_path = SIONNA_MUNICH_LEGACY_PATH if legacy_path is None else legacy_path
    return ka_path if os.path.exists(ka_path) else legacy_path


# THE single path SionnaMunichIterator's default (no legacy-link override) branch
# consults -- resolved ONCE here (ka-preferred, else legacy), so it stays a plain,
# `monkeypatch.setattr`-able module attribute. This is a pre-existing test contract
# (tests/test_blocks.py monkeypatches this exact name to point at a temp multi-link pkl
# and relies on it being the SOLE authority `SionnaMunichIterator` consults for the
# default case -- checking `SIONNA_MUNICH_KA_PATH.exists()` at call time instead, as an
# earlier version of this file did, silently loaded the real generated munich_ka.pkl over
# the monkeypatched path and broke that contract, since the real file exists on disk).
SIONNA_MUNICH_PATH = _resolve_munich_default_path()
# Special `link` value that selects the legacy 3.5 GHz file through the same
# `SionnaMunichIterator(link=...)`/`SionnaEnvironmentBlock('munich', link=...)` call site
# -- lets a caller pick the legacy artifact without a second scenario name.
MUNICH_LEGACY_LINK = 'munich_legacy_3p5ghz'
# ...and the same mechanism for the swept file. A FILE selector, exactly like
# MUNICH_LEGACY_LINK: `SionnaEnvironmentBlock('munich', link=MUNICH_LOSWEEP_LINK)`.
MUNICH_LOSWEEP_LINK = 'munich_ka_losweep'


# Factories forward an optional `link` selector to SionnaIterator so a multi-link pkl can
# be addressed explicitly. They take no positional args, so existing `SionnaMunichIterator()`
# / `SionnaEtoileIterator()` call sites keep working (link=None -> first link / single array).
def SionnaEtoileIterator(link=None):
    return SionnaIterator(SIONNA_ETOILE_PATH, link=link)


def SionnaMunichIterator(link=None):
    # `link=MUNICH_LEGACY_LINK` is a FILE selector here, not a pkl-internal link name
    # (the legacy pkl is a bare ndarray with no links at all) -- it exists so the legacy
    # 3.5 GHz artifact stays reachable through 'munich' without a second scenario name.
    # Any OTHER link value (including None) is pkl-internal link semantics, forwarded
    # untouched to SionnaIterator against SIONNA_MUNICH_PATH -- this must NOT do its own
    # ka-vs-legacy existence check (see that attribute's docstring above).
    if link == MUNICH_LEGACY_LINK:
        return SionnaIterator(SIONNA_MUNICH_LEGACY_PATH, link=None)
    if link == MUNICH_LOSWEEP_LINK:
        # Same FILE-selector branch as the legacy one above, for the swept-line-of-sight
        # Ka trace (Thrust 3). `link=None` inside it: that file is written by the same
        # generator as munich_ka.pkl and carries one link, whose name is not this token.
        return SionnaIterator(SIONNA_MUNICH_LOSWEEP_PATH, link=None)
    return SionnaIterator(SIONNA_MUNICH_PATH, link=link)

