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

parser.add_argument ("--beam-map", dest="bmap",default='TRUE',
                     choices=TrueFalseOverwrite,
                     help="compute the BEAM_MAP products [TRUE]");

parser.add_argument ("--preproc", dest="preproc",default='TRUE',
                     choices=TrueFalseOverwrite,
                     help="compute the PREPROC products [TRUE]");

parser.add_argument ("--output-dir", dest="outputDir",default='./reduced/',
                     help="output directories for product");

parser.add_argument ("--max-file", dest="maxFile",default=None,
                     help="maximum nuber of file to load to build"
                          "product (speed-up for tests)");

parser.add_argument ("--delta-time", dest="dTime",default=300,
                     help="maximum time between files to be groupped (s) [300]");

# Parse argument
argopt = parser.parse_args();

#
# Initialisation
#

# Output
outputDir = argopt.outputDir;
dTime = float(argopt.dTime);

# Format maxFile
mf = argopt.maxFile;
if mf is not None:
    mrx.log.warning ('--max-file set to '+mf+', wont use all data!!');
    mf = int(mf);

# Define setup keys
keys = mrx.setup.detector + mrx.setup.instrument;

# Get all RAW files from current dir 
hdrs_raw = mrx.headers.loaddir ('./');

#
# Compute BACKGROUND_MEAN
#

if argopt.background != 'FALSE':
    
    # Group backgrounds
    gps = mrx.headers.group (hdrs_raw, 'BACKGROUND', delta=dTime);
    overwrite = (argopt.background == 'OVERWRITE');

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
            if argopt.debug == 'TRUE': raise;
        finally:
            mrx.log.closeFile ();



#
# Compute BEAMi_MAP
#

if argopt.bmap != 'FALSE':
        
    # Read all calibration products
    hdrs_calib = mrx.headers.loaddir (outputDir);
    
    # Group all BEAMi
    gps = mrx.headers.group (hdrs_raw, 'BEAM', delta=dTime);
    overwrite = (argopt.bmap == 'OVERWRITE');

    # Compute all 
    for i,gp in enumerate(gps):
        try:
            mrx.log.info ('Compute BEAM_MAP {0} over {1} '.format(i+1,len(gps)));

            name = gp[0]['FILETYPE'].lower()+'map';
            output = mrx.files.output (outputDir, gp[0], name);
            if os.path.exists (output+'.fits') and overwrite is False:
                mrx.log.info ('Product already exists');
                continue;
            
            mrx.log.setFile (output+'.log');
            
            bkg = mrx.headers.assoc (gp[0], hdrs_calib, 'BACKGROUND_MEAN',
                                     keys, which='closest', required=1);
            
            mrx.compute_beammap (gp[0:mf], bkg, output=output);
            
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
    hdrs_calib = mrx.headers.loaddir (outputDir);

    # Group all DATA
    gps = mrx.headers.group (hdrs_raw, 'DATA', delta=dTime);
    overwrite = (argopt.preproc == 'OVERWRITE');

    # Compute 
    for i,gp in enumerate(gps):
        try:
            mrx.log.info ('Compute preproc {0} over {1} '.format(i+1,len(gps)));
            
            output = mrx.files.output (outputDir, gp[0], 'preproc');
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
            
            mrx.compute_preproc (gp[0:mf], bkg, pmaps, output=output);
            
        except Exception as exc:
            mrx.log.error ('Cannot compute preproc: '+str(exc));
            if argopt.debug == 'TRUE': raise;
        finally:
            mrx.log.closeFile ();
            
            

