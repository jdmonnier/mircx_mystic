#! /usr/bin/env python                                                          
# -*- coding: iso-8859-15 -*-                                                   

import mircx_pipeline as mrx
import glob

# 
overwrite = False;
keys = mrx.setup.detector + mrx.setup.instrument;

# Get all RAW files from current dir 
hdrs_raw = mrx.headers.loaddir ('./');

#
# Compute BACKGROUND_REDUCED
#

# Group backgrounds
gps = mrx.headers.group (hdrs_raw, 'BACKGROUND');

# Compute all backgrounds
for i,gp in enumerate(gps):
    mrx.log.info ('Compute background {0} over {1} '.format(i+1,len(gps)));
    mrx.compute_background (gp, overwrite=overwrite);

#
# Compute PIXMAP
#

# Group all FOREGROUND
gps = mrx.headers.group (hdrs_raw, 'FOREGROUND');

# Read all calibration products
hdrs_calib = mrx.headers.loaddir ('./calib/');

# Compute all window
for i,gp in enumerate(gps):
    mrx.log.info ('Compute window {0} over {1} '.format(i+1,len(gps)));
    bkg = mrx.headers.assoc (gp[0], hdrs_calib, 'BACKGROUND_REDUCED',
                          keys, which='closest', required=1);
    mrx.compute_windows (gp, bkg[0], overwrite=overwrite);

#
# Compute PREPROC
#

# Group all DATA
gps = mrx.headers.group (hdrs_raw, 'DATA');

# Read all calibration products
hdrs_calib = mrx.headers.loaddir ('./calib/');

# Compute 
for i,gp in enumerate(gps):
    mrx.log.info ('Compute preproc {0} over {1} '.format(i+1,len(gps)));
    bkg = mrx.headers.assoc (gp[0], hdrs_calib, 'BACKGROUND_REDUCED',
                             keys, which='closest', required=1);
    win = mrx.headers.assoc (gp[0], hdrs_calib, 'PIXMAP',
                             keys, which='closest', required=1);
    mrx.compute_preproc (gp,bkg[0],win[0], overwrite=overwrite);

