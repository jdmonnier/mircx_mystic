#! /usr/bin/env python                                                          
# -*- coding: iso-8859-15 -*-                                                   

import mircx_pipeline as mrx
import argparse
import glob
import os

#
# Implement options
#

# Describe the script
description = \
"""
description:
  Run the mircx test pipeline
  Output the results in calib/ and reduced/
"""

epilog = \
"""
examples:
  cd /path/to/my/data/
  mirx_preproc.py --background=FALSE --output-dir=./mytest/
"""

parser = argparse.ArgumentParser (description=description, epilog=epilog,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter);

TrueFalse = ['TRUE','FALSE'];
TrueFalseOverwrite = ['TRUE','FALSE','OVERWRITE'];

parser.add_argument ("--debug", dest="debug",default='FALSE',
                     choices=TrueFalse,
                     help="stop or error");

parser.add_argument ("--output-dir", dest="outputDir",default='./reduced/',type=str,
                     help="output directories for product");

parser.add_argument ("--max-file", dest="max_file",default=300,type=int,
                     help="maximum nuber of file to load to build "
                          "product (speed-up for tests)");

parser.add_argument ("--delta-time", dest="delta",default=300,type=float,
                     help="maximum time between files to be groupped (s)");

parser.add_argument ("--background", dest="background",default='FALSE',
                     choices=TrueFalseOverwrite,
                     help="compute the BACKGROUND products");

parser.add_argument ("--beam-map", dest="bmap",default='FALSE',
                     choices=TrueFalseOverwrite,
                     help="compute the BEAM_MAP products");

parser.add_argument ("--preproc", dest="preproc",default='FALSE',
                     choices=TrueFalseOverwrite,
                     help="compute the PREPROC products");

parser.add_argument ("--rts", dest="rts",default='FALSE',
                     choices=TrueFalseOverwrite,
                     help="compute the RTS products");

parser.add_argument ("--vis", dest="vis",default='FALSE',
                     choices=TrueFalseOverwrite,
                     help="compute the VIS products");

#
# Initialisation
#

# Parse argument
argopt = parser.parse_args ();

# Define setup keys
keys = mrx.setup.detector + mrx.setup.instrument;

# Get all RAW files from current dir
if argopt.background != 'FALSE' or argopt.preproc != 'FALSE':
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
            mrx.log.info ('Compute BACKGROUND_MEAN {0} over {1} '.format(i+1,len(gps)));
            
            output = mrx.files.output (argopt.outputDir, gp[0], 'bkg');
            if os.path.exists (output+'.fits') and overwrite is False:
                mrx.log.info ('Product already exists');
                continue;
                
            mrx.log.setFile (output+'.log');

            mrx.compute_background (gp[0:argopt.max_file], output=output);
            
        except Exception as exc:
            mrx.log.error ('Cannot compute BACKGROUND_MEAN: '+str(exc));
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
            
            mrx.compute_beammap (gp[0:argopt.max_file], bkg, output=output);
            
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
            mrx.log.info ('Compute PREPROC {0} over {1} '.format(i+1,len(gps)));
            
            output = mrx.files.output (argopt.outputDir, gp[0], 'preproc');
            if os.path.exists (output+'.fits') and overwrite is False:
                mrx.log.info ('Product already exists');
                continue;

            mrx.log.setFile (output+'.log');
                
            bkg  = mrx.headers.assoc (gp[0], hdrs_calib, 'BACKGROUND_MEAN',
                                    keys, which='closest', required=1);

            bmaps = [];
            for i in range(1,7):
                tmp = mrx.headers.assoc (gp[0], hdrs_calib, 'BEAM%i_MAP'%i,
                                         keys, which='best', required=1);
                bmaps.extend(tmp);
            
            mrx.compute_preproc (gp[0:argopt.max_file], bkg, bmaps, output=output);
            
        except Exception as exc:
            mrx.log.error ('Cannot compute PREPROC: '+str(exc));
            if argopt.debug == 'TRUE': raise;
        finally:
            mrx.log.closeFile ();
            

#
# Compute RTS
#

if argopt.rts != 'FALSE':

    # Read all calibration products
    hdrs_calib = mrx.headers.loaddir (argopt.outputDir);

    # Group all DATA
    gps = mrx.headers.group (hdrs_calib, 'DATA_PREPROC', delta=0);
    overwrite = (argopt.rts == 'OVERWRITE');

    # Compute 
    for i,gp in enumerate(gps):
        try:
            mrx.log.info ('Compute RTS {0} over {1} '.format(i+1,len(gps)));
            
            output = mrx.files.output (argopt.outputDir, gp[0], 'rts');
            if os.path.exists (output+'.fits') and overwrite is False:
                mrx.log.info ('Product already exists');
                continue;

            mrx.log.setFile (output+'.log');
            
            bmaps = [];
            for i in range(1,7):
                tmp = mrx.headers.assoc (gp[0], hdrs_calib, 'BEAM%i_MAP'%i,
                                         keys, which='best', required=1);
                bmaps.extend(tmp);
            
            mrx.compute_rts (gp, bmaps, output=output);

        except Exception as exc:
            mrx.log.error ('Cannot compute RTS: '+str(exc));
            if argopt.debug == 'TRUE': raise;
        finally:
            mrx.log.closeFile ();
            
#
# Compute VIS
#

if argopt.vis != 'FALSE':

    # Read all calibration products
    hdrs_calib = mrx.headers.loaddir (argopt.outputDir);

    # Group all DATA
    gps = mrx.headers.group (hdrs_calib, 'RTS', delta=0);
    overwrite = (argopt.vis == 'OVERWRITE');

    # Compute 
    for i,gp in enumerate(gps):
        try:
            mrx.log.info ('Compute VIS {0} over {1} '.format(i+1,len(gps)));
            
            output = mrx.files.output (argopt.outputDir, gp[0], 'vis');
            if os.path.exists (output+'.fits') and overwrite is False:
                mrx.log.info ('Product already exists');
                continue;

            mrx.log.setFile (output+'.log');

            # for nc in [0,0.5,1,2,4,8,16,32,64,128]:
            for nc in [0.5]:
                mrx.compute_vis (gp, output=output+'_c%04i'%int(nc*10), ncoher=nc);

        except Exception as exc:
            mrx.log.error ('Cannot compute VIS: '+str(exc));
            if argopt.debug == 'TRUE': raise;
        finally:
            mrx.log.closeFile ();
            
                                    

