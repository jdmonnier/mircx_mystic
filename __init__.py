"""
Module to reduce the MIRCX MYSTIC data
"""

# General modules
from . import headers, setup, files, oifits, signal, catalog;

# High levels routines to process RAW
# data into BEAMMAP and PREPROC
from .preproc import compute_background
from .preproc import compute_beam_map
from .preproc import compute_beam_profile
from .preproc import compute_preproc


# High levels routines to extract vis
# from BEAMMAP and PREPROC data
from .selection import compute_selection
from .bbias import compute_bbias_coeff
from .vis import compute_speccal
from .vis import compute_rts
from .vis import compute_vis

# High levels routines to calibrate 
from .viscalib import compute_all_viscalib

# Revision number
from .version import revision, info
info();
