#! /usr/bin/env python                                                          
# -*- coding: iso-8859-15 -*-                                                   

import mircx_pipeline as mrx
import glob
import argparse

usage = """
description:
  Run the mircx test pipeline
  Output the results in calib/ and reduced/
"""

examples = """
examples:
  cd /path/to/my/data/
  mirx_preproc.py --background=FALSE
 
"""

#
# Implement options
#

parser = argparse.ArgumentParser (description=usage, epilog=examples,
                                 formatter_class=argparse.RawDescriptionHelpFormatter);
TrueFalse = ['TRUE','FALSE'];
TrueFalseOverwrite = ['TRUE','FALSE','OVERWRITE'];

parser.add_argument ("--background", dest="background",default='TRUE',choices=TrueFalseOverwrite,
                    help="compute the BACKGROUND products [TRUE]");

parser.add_argument ("--pixmap", dest="pixmap",default='TRUE',choices=TrueFalseOverwrite,
                    help="compute the PIXMAP products [TRUE]");

parser.add_argument ("--preproc", dest="preproc",default='TRUE',choices=TrueFalseOverwrite,
                    help="compute the PREPROC products [TRUE]");

# Parse argument
argoptions = parser.parse_args();

dirCalib   = './calib/';
dirReduced = './reduced/';

#
# Initialisation
#


# Define setup keys
keys = mrx.setup.detector + mrx.setup.instrument;

# Get all RAW files from current dir 
hdrs_raw = mrx.headers.loaddir ('./');

#
# Compute BACKGROUND_REDUCED
#

if argoptions.background != 'FALSE':
    
    # Group backgrounds
    gps = mrx.headers.group (hdrs_raw, 'BACKGROUND');
    overwrite = (argoptions.background == 'OVERWRITE');

    # Compute all backgrounds
    for i,gp in enumerate(gps):
        mrx.log.info ('Compute background {0} over {1} '.format(i+1,len(gps)));
        mrx.compute_background (gp, overwrite=overwrite);

#
# Compute PIXMAP
#

if argoptions.pixmap != 'FALSE':

    # Group all FOREGROUND
    gps = mrx.headers.group (hdrs_raw, 'FOREGROUND');
    overwrite = (argoptions.pixmap == 'OVERWRITE');

    # Read all calibration products
    hdrs_calib = mrx.headers.loaddir (dirCalib);

    # Compute all window
    for i,gp in enumerate(gps):
        mrx.log.info ('Compute window {0} over {1} '.format(i+1,len(gps)));
        bkg = mrx.headers.assoc (gp[0], hdrs_calib, 'BACKGROUND_REDUCED',
                                 keys, which='closest', required=1);
        mrx.compute_windows (gp, bkg[0], overwrite=overwrite);

#
# Compute PREPROC
#

if argoptions.preproc != 'FALSE':

    # Group all DATA
    gps = mrx.headers.group (hdrs_raw, 'DATA');
    overwrite = (argoptions.preproc == 'OVERWRITE');

    # Read all calibration products
    hdrs_calib = mrx.headers.loaddir (dirCalib);

    # Compute 
    for i,gp in enumerate(gps):
        mrx.log.info ('Compute preproc {0} over {1} '.format(i+1,len(gps)));
        bkg = mrx.headers.assoc (gp[0], hdrs_calib, 'BACKGROUND_REDUCED',
                                keys, which='closest', required=1);
        win = mrx.headers.assoc (gp[0], hdrs_calib, 'PIXMAP',
                                keys, which='closest', required=1);
        mrx.compute_preproc (gp,bkg[0],win[0], overwrite=overwrite);

