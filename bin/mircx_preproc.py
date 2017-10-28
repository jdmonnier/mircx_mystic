#! /usr/bin/env python                                                          
# -*- coding: iso-8859-15 -*-                                                   

import mircx_pipeline as mrx
import glob
import os

#
# Implement options
#

# Describe the script
mrx.batch.parser.description = \
"""
description:
  Run the mircx test pipeline
  Output the results in calib/ and reduced/
"""

mrx.batch.parser.epilog = \
"""
examples:
  cd /path/to/my/data/
  mirx_preproc.py --background=FALSE --output-dir=./mytest/
"""

# Parse arguments
argopt = mrx.batch.parser.parse_args ();

#
# Initialisation
#

# Define setup keys
keys = mrx.setup.detector + mrx.setup.instrument;

# Get all RAW files from current dir 
hdrs_raw = mrx.headers.loaddir ('./');

#
# Compute BACKGROUND_MEAN
#

if argopt.background != 'FALSE':
    
    # Group backgrounds
    gps = mrx.headers.group (hdrs_raw, 'BACKGROUND', delta=argopt.delta);
    overwrite = (argopt.background == 'OVERWRITE');

    # Compute all backgrounds
    for i,gp in enumerate(gps):
        try:
            mrx.log.info ('Compute background {0} over {1} '.format(i+1,len(gps)));
            
            output = mrx.files.output (argopt.outputDir, gp[0], 'bkg');
            if os.path.exists (output+'.fits') and overwrite is False:
                mrx.log.info ('Product already exists');
                continue;
                
            mrx.log.setFile (output+'.log');

            mrx.compute_background (gp[0:argopt.mf], output=output);
            
        except Exception as exc:
            mrx.log.error ('Cannot compute background: '+str(exc));
            if argopt.debug == 'TRUE': raise;
        finally:
            mrx.log.closeFile ();



#
# Compute BEAMi_MAP
#

if argopt.bmap != 'FALSE':
        
    # Read all calibration products
    hdrs_calib = mrx.headers.loaddir (argopt.outputDir);
    
    # Group all BEAMi
    gps = mrx.headers.group (hdrs_raw, 'BEAM', delta=argopt.delta);
    overwrite = (argopt.bmap == 'OVERWRITE');

    # Compute all 
    for i,gp in enumerate(gps):
        try:
            mrx.log.info ('Compute BEAM_MAP {0} over {1} '.format(i+1,len(gps)));

            name = gp[0]['FILETYPE'].lower()+'map';
            output = mrx.files.output (argopt.outputDir, gp[0], name);
            if os.path.exists (output+'.fits') and overwrite is False:
                mrx.log.info ('Product already exists');
                continue;
            
            mrx.log.setFile (output+'.log');
            
            bkg = mrx.headers.assoc (gp[0], hdrs_calib, 'BACKGROUND_MEAN',
                                     keys, which='closest', required=1);
            
            mrx.compute_beammap (gp[0:argopt.mf], bkg, output=output);
            
        except Exception as exc:
            mrx.log.error ('Cannot compute BEAM_MAP: '+str(exc));
            if argopt.debug == 'TRUE': raise;
        finally:
            mrx.log.closeFile ();
        

    

#
# Compute PREPROC
#

if argopt.preproc != 'FALSE':

    # Read all calibration products
    hdrs_calib = mrx.headers.loaddir (argopt.outputDir);

    # Group all DATA
    gps = mrx.headers.group (hdrs_raw, 'DATA', delta=argopt.delta);
    overwrite = (argopt.preproc == 'OVERWRITE');

    # Compute 
    for i,gp in enumerate(gps):
        try:
            mrx.log.info ('Compute preproc {0} over {1} '.format(i+1,len(gps)));
            
            output = mrx.files.output (argopt.outputDir, gp[0], 'preproc');
            if os.path.exists (output+'.fits') and overwrite is False:
                mrx.log.info ('Product already exists');
                continue;

            mrx.log.setFile (output+'.log');
                
            bkg  = mrx.headers.assoc (gp[0], hdrs_calib, 'BACKGROUND_MEAN',
                                    keys, which='closest', required=1);

            pmaps = [];
            for i in range(1,7):
                tmp = mrx.headers.assoc (gp[0], hdrs_calib, 'BEAM%i_MAP'%i,
                                         keys, which='closest', required=1);
                pmaps.extend(tmp);
            
            mrx.compute_preproc (gp[0:argopt.mf], bkg, pmaps, output=output);
            
        except Exception as exc:
            mrx.log.error ('Cannot compute preproc: '+str(exc));
            if argopt.debug == 'TRUE': raise;
        finally:
            mrx.log.closeFile ();
            
            

