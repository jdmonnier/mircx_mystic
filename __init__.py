"""
  Module to reduce the MIRCX data
"""

from . import headers, setup, files, batch, oifits


from .mircx import compute_background
from .mircx import compute_beammap
from .mircx import compute_preproc
from .mircx import compute_rts
from .mircx import compute_vis

from .version import revision, info
info ();
    
