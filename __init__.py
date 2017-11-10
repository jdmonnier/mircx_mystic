"""
  Module to reduce the MIRCX data
"""

# General modules
from . import headers, setup, files, batch, oifits, signal

# High levels routines to process detector
from .preproc import compute_background
from .preproc import compute_beammap
from .preproc import compute_preproc

# High levels routines to extract vis
from .vis import compute_speccal
from .vis import compute_rts
from .vis import compute_vis

# Revision number
from .version import revision, info
info();
