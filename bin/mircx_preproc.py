#! /usr/bin/env python                                                          
# -*- coding: iso-8859-15 -*-                                                   

import mircx_pipeline as mrx
import glob
import argparse
import os

usage = """
description:
  Run the mircx test pipeline
  Output the results in calib/ and reduced/
"""

examples = """
examples:
  cd /path/to/my/data/
  mirx_preproc.py --background=FALSE --output-dir=./mytest/
 
"""

#
# Implement options
#

parser = argparse.ArgumentParser (description=usage, epilog=examples,
                                 formatter_class=argparse.RawDescriptionHelpFormatter);
TrueFalse = ['TRUE','FALSE'];
TrueFalseOverwrite = ['TRUE','FALSE','OVERWRITE'];

parser.add_argument ("--debug", dest="debug",default='FALSE',
                     choices=TrueFalse,
                     help="stop or error [TRUE]");

parser.add_argument ("--background", dest="background",default='TRUE',
                     choices=TrueFalseOverwrite,
                     help="compute the BACKGROUND products [TRUE]");

parser.add_argument ("--pixmap", dest="pixmap",default='TRUE',
                     choices=TrueFalseOverwrite,
                     help="compute the PIXMAP products [TRUE]");

parser.add_argument ("--preproc", dest="preproc",default='TRUE',
                     choices=TrueFalseOverwrite,
                     help="compute the PREPROC products [TRUE]");

parser.add_argument ("--output-dir", dest="outputDir",default='./reduced/',
                     help="output directories for product");

parser.add_argument ("--max-file", dest="maxFile",default=None,
                     help="maximum nuber of file to load to build"
                          "product (speed-up for tests)");

# Parse argument
argoptions = parser.parse_args();

#
# Initialisation
#

# Output
outputDir = argoptions.outputDir;

# Format maxFile
mf = argoptions.maxFile;
if mf is not None:
    mrx.log.warning ('--max-file set to '+mf+', wont use all data!!');
    mf = int(mf);

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
        try:
            mrx.log.info ('Compute background {0} over {1} '.format(i+1,len(gps)));
            
            output = mrx.files.output (outputDir, gp[0], 'bkg');
            if os.path.exists (output+'.fits') and overwrite is False:
                mrx.log.info ('Product already exists');
                continue;
                
            mrx.log.setFile (output+'.log');

            mrx.compute_background (gp[0:mf], output=output);
            
        except Exception as exc:
            mrx.log.error ('Cannot compute background: '+str(exc));
            if argoptions.debug == 'TRUE': raise;
        finally:
            mrx.log.closeFile ();


#
# Compute PIXMAP
#

if argoptions.pixmap != 'FALSE':

    # Group all FOREGROUND
    gps = mrx.headers.group (hdrs_raw, 'FOREGROUND');
    overwrite = (argoptions.pixmap == 'OVERWRITE');

    # Read all calibration products
    hdrs_calib = mrx.headers.loaddir (outputDir);

    # Compute all pixmap
    for i,gp in enumerate(gps):
        try:
            mrx.log.info ('Compute pixmap {0} over {1} '.format(i+1,len(gps)));

            output = mrx.files.output (outputDir, gp[0], 'pixmap');
            if os.path.exists (output+'.fits') and overwrite is False:
                mrx.log.info ('Product already exists');
                continue;
            
            mrx.log.setFile (output+'.log');
            
            bkg = mrx.headers.assoc (gp[0], hdrs_calib, 'BACKGROUND_REDUCED',
                                     keys, which='closest', required=1);
            
            mrx.compute_pixmap (gp[0:mf], bkg, output=output);
            
        except Exception as exc:
            mrx.log.error ('Cannot compute pixmap: '+str(exc));
            if argoptions.debug == 'TRUE': raise;
        finally:
            mrx.log.closeFile ();

#
# Compute PREPROC
#

if argoptions.preproc != 'FALSE':

    # Group all DATA
    gps = mrx.headers.group (hdrs_raw, 'DATA');
    overwrite = (argoptions.preproc == 'OVERWRITE');

    # Read all calibration products
    hdrs_calib = mrx.headers.loaddir (outputDir);

    # Compute 
    for i,gp in enumerate(gps):
        try:
            mrx.log.info ('Compute preproc {0} over {1} '.format(i+1,len(gps)));
            
            output = mrx.files.output (outputDir, gp[0], 'preproc');
            if os.path.exists (output+'.fits') and overwrite is False:
                mrx.log.info ('Product already exists');
                continue;

            mrx.log.setFile (output+'.log');
                
            bkg  = mrx.headers.assoc (gp[0], hdrs_calib, 'BACKGROUND_REDUCED',
                                    keys, which='closest', required=1);
            
            pmap = mrx.headers.assoc (gp[0], hdrs_calib, 'PIXMAP',
                                    keys, which='closest', required=1);
            
            mrx.compute_preproc (gp[0:mf], bkg, pmap, output=output);
            
        except Exception as exc:
            mrx.log.error ('Cannot compute preproc: '+str(exc));
            if argoptions.debug == 'TRUE': raise;
        finally:
            mrx.log.closeFile ();
            
            

