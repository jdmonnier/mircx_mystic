#! /usr/bin/env python                                                          
# -*- coding: iso-8859-15 -*-                                                   

import mircx_pipeline as mrx
import argparse, glob, os

from mircx_pipeline import log, setup;
from astropy.io import fits as pyfits;

#
# Implement options
#

# Describe the script
description = \
"""
description:
 Plot a report of data reduced by the pipeline.
 Should be run in a directory where the OIFITS
 are stored or use the option --oifits-dir

"""

epilog = \
"""
examples:

   cd /my/reduced/data/oifits/
  mircx_report.py


"""

parser = argparse.ArgumentParser (description=description, epilog=epilog,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 add_help=True);


parser.add_argument ("--oifits-dir", dest="oifits_dir",default='./',type=str,
                     help="directory of products [%(default)s]");


#
# Initialisation
#

# Parse argument
argopt = parser.parse_args ();

# Verbose
elog = log.trace ('mircx_report');

# List inputs
files = glob.glob (argopt.oifits_dir+'/mircx*oifits.fit*');
nf = len (files);

# Init variables for plots. Fill them with NaN so
# that missing points will not be plotted
nb = 15;

vis2  = np.zeros ((nb,nf)) * np.nan;
mjd   = np.zeros (nf) * np.nan;
flux  = np.zeros ((nb,nf)) * np.nan;
Hmag  = np.zeros ((nb,nf)) * np.nan;
lbd   = np.zeros (nf) * np.nan;
diam  = np.zeros (nf) * np.nan;
uv    = np.zeros ((nb,nf)) * np.nan;


#
# Loop on files to load data
#

for f in files:
    
    try:
        
        # Load data
        hdr = pyfits.getheader (f);
        oivis  = pyfits.getdata (f, 'OI_VIS2');
        oiflux = pyfits.getdata (f, 'OI_FLUX');
        
        # Set them into variables

    # Handle exceptions so that the script won't crash
    # oif a file is corrupted
    except Exception as exc:
        log.error ('Cannot load OIFITS: '+str(exc));
        if argopt.debug == 'TRUE': raise;

            
#
# Plots the trends
#



