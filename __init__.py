"""
  Module to reduce the MIRCX data
"""

from . import headers, setup, files, batch


from .mircx import compute_background
from .mircx import compute_beammap
from .mircx import compute_preproc

from .version import revision, info
info ();
    
