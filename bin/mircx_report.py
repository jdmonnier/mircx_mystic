#! /usr/bin/env python                                                          
# -*- coding: iso-8859-15 -*-                                                   

import mircx_pipeline as mrx;
import argparse, glob, os;
import numpy as np;

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

TrueFalse = ['TRUE','FALSE'];

parser = argparse.ArgumentParser (description=description, epilog=epilog,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 add_help=True);

parser.add_argument ("--oifits-dir", dest="oifits_dir",default='./',type=str,
                     help="directory of products [%(default)s]");

parser.add_argument ("--debug", dest="debug",default='FALSE',
                     choices=TrueFalse,
                     help="stop on error [%(default)s]");

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
nt = 6;

vis2  = np.zeros ((nb,nf)) * np.nan;
u     = np.zeros ((nb,nf)) * np.nan;
v     = np.zeros ((nb,nf)) * np.nan;
flux  = np.zeros ((nt,nf)) * np.nan;
mjd   = np.zeros (nf) * np.nan;
Hmag  = np.zeros (nf) * np.nan;
lbd   = np.zeros (nf) * np.nan;
diam  = np.zeros (nf) * np.nan;


#
# Loop on files to load data
#

for f,file in enumerate(files):
    
    try:
        log.info ('Load file '+file);
        
        # Load header data
        hdr = pyfits.getheader (file);
        
        # Load OI_WAVELENGTH
        eff_wave = pyfits.getdata (file, 'OI_WAVELENGTH')['EFF_WAVE'];
        ny = len (eff_wave)/2;
        lbd[f] = eff_wave[ny];
        
        # Load OI_VIS2 and OI_FLUX
        vis2[:,f] = pyfits.getdata (file, 'OI_VIS2')['VIS2DATA'][:,ny];
        u[:,f]    = pyfits.getdata (file, 'OI_VIS2')['UCOORD'];
        v[:,f]    = pyfits.getdata (file, 'OI_VIS2')['VCOORD'];
        flux[:,f] = pyfits.getdata (file, 'OI_FLUX')['FLUXDATA'][:,ny];

        # Get the Hmag and Diameter from catalog ?
        # ...
        
    # Handle exceptions so that the script won't
    # crash if a file is corrupted
    except Exception as exc:
        log.error ('Cannot load OIFITS: '+str(exc));
        if argopt.debug == 'TRUE': raise;

            
#
# Plots the trends
#

log.info ('Figures');



