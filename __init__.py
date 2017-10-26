"""
  Module to reduce the MIRCX data
"""

from . import headers, setup, files

from .mircx import compute_background
from .mircx import compute_fringemap
from .mircx import compute_beammap
from .mircx import compute_preproc
from .mircx import compute_snr
