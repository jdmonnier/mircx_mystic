#! /usr/bin/env python                                                          
# -*- coding: iso-8859-15 -*-                                                   

import mircx_pipeline as mrx
import argparse
import glob
import os

from mircx_pipeline import log, setup;

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
                     help="maximum number of file to load to build "
                          "product (speed-up for tests)");

parser.add_argument ("--delta-time", dest="delta",default=300,type=float,
                     help="maximum time between consecutive files to be groupped (s)");

parser.add_argument ("--Delta-time", dest="Delta",default=300,type=float,
                     help="maximum time between first and last files to be groupped (s)");

parser.add_argument ("--continuous-time", dest="cont",default='TRUE',
                     choices=TrueFalse,
                     help="create a new group with different filetype interleaved");

parser.add_argument ("--background", dest="background",default='FALSE',
                     choices=TrueFalseOverwrite,
                     help="compute the BACKGROUND products");

parser.add_argument ("--beam-map", dest="bmap",default='FALSE',
                     choices=TrueFalseOverwrite,
                     help="compute the BEAM_MAP products");

parser.add_argument ("--preproc", dest="preproc",default='FALSE',
                     choices=TrueFalseOverwrite,
                     help="compute the PREPROC products");

parser.add_argument ("--spec-cal", dest="speccal",default='FALSE',
                     choices=TrueFalseOverwrite,
                     help="compute the SPEC_CAL products");

parser.add_argument ("--rts", dest="rts",default='FALSE',
                     choices=TrueFalseOverwrite,
                     help="compute the RTS products");

parser.add_argument ("--vis", dest="vis",default='FALSE',
                     choices=TrueFalseOverwrite,
                     help="compute the VIS products");

parser.add_argument ("--vis-calibrated", dest="viscalib",default='FALSE',
                     choices=TrueFalseOverwrite,
                     help="compute the VIS_CALIBRATED products");

parser.add_argument ("--calibrators", dest="calibrators",default='name,diam,err',
                     type=str, help="list of calibration star with diameters");

#
# Initialisation
#

# Parse argument
argopt = parser.parse_args ();

# Verbose
elog = log.trace ('mircx_preproc');

# Get all RAW files from current dir
if argopt.background != 'FALSE' or \
   argopt.bmap != 'FALSE' or \
   argopt.preproc != 'FALSE':
    hdrs_raw = mrx.headers.loaddir ('./');

#
# Compute BACKGROUND_MEAN
#

if argopt.background != 'FALSE':
    
    # Group backgrounds
    gps = mrx.headers.group (hdrs_raw, 'BACKGROUND', delta=argopt.delta, Delta=argopt.Delta,
                             keys=setup.detwin+setup.detmode+setup.insmode,
                             continuous=argopt.cont);
    overwrite = (argopt.background == 'OVERWRITE');

    # Compute all backgrounds
    for i,gp in enumerate(gps):
        try:
            log.info ('Compute BACKGROUND_MEAN {0} over {1} '.format(i+1,len(gps)));
            
            output = mrx.files.output (argopt.outputDir, gp[0], 'bkg');
            if os.path.exists (output+'.fits') and overwrite is False:
                log.info ('Product already exists');
                continue;
                
            log.setFile (output+'.log');

            mrx.compute_background (gp[0:argopt.max_file], output=output);
            
        except Exception as exc:
            log.error ('Cannot compute BACKGROUND_MEAN: '+str(exc));
            if argopt.debug == 'TRUE': raise;
        finally:
            log.closeFile ();



#
# Compute BEAMi_MAP
#

if argopt.bmap != 'FALSE':
        
    # Read all calibration products
    hdrs_calib = mrx.headers.loaddir (argopt.outputDir);
    
    # Group all BEAMi
    gps = mrx.headers.group (hdrs_raw, 'BEAM.*', delta=argopt.delta, Delta=argopt.Delta,
                             keys=setup.detwin+setup.detmode+setup.insmode,
                             continuous=argopt.cont);
    overwrite = (argopt.bmap == 'OVERWRITE');

    # Compute all 
    for i,gp in enumerate(gps):
        try:
            log.info ('Compute BEAM_MAP {0} over {1} '.format(i+1,len(gps)));

            name = gp[0]['FILETYPE'].lower()+'map';
            output = mrx.files.output (argopt.outputDir, gp[0], name);
            if os.path.exists (output+'.fits') and overwrite is False:
                log.info ('Product already exists');
                continue;
            
            log.setFile (output+'.log');
            
            bkg = mrx.headers.assoc (gp[0], hdrs_calib, 'BACKGROUND_MEAN',
                                     keys=setup.detwin+setup.detmode+setup.insmode,
                                     which='closest', required=1);
            
            mrx.compute_beammap (gp[0:argopt.max_file], bkg, output=output);
            
        except Exception as exc:
            log.error ('Cannot compute BEAM_MAP: '+str(exc));
            if argopt.debug == 'TRUE': raise;
        finally:
            log.closeFile ();
        
#
# Compute PREPROC
#

if argopt.preproc != 'FALSE':

    # Read all calibration products
    hdrs_calib = mrx.headers.loaddir (argopt.outputDir);

    # Group all DATA
    gps = mrx.headers.group (hdrs_raw, 'DATA', delta=argopt.delta, Delta=argopt.Delta,
                             keys=setup.detwin+setup.detmode+setup.insmode,
                             continuous=argopt.cont);
    overwrite = (argopt.preproc == 'OVERWRITE');

    # Compute 
    for i,gp in enumerate(gps):
        try:
            log.info ('Compute PREPROC {0} over {1} '.format(i+1,len(gps)));
            
            output = mrx.files.output (argopt.outputDir, gp[0], 'preproc');
            if os.path.exists (output+'.fits') and overwrite is False:
                log.info ('Product already exists');
                continue;

            log.setFile (output+'.log');
                
            bkg  = mrx.headers.assoc (gp[0], hdrs_calib, 'BACKGROUND_MEAN',
                                     keys=setup.detwin+setup.detmode+setup.insmode,
                                     which='closest', required=1);

            # Associate MAP
            bmaps = [];
            for i in range(1,7):
                tmp = mrx.headers.assoc (gp[0], hdrs_calib, 'BEAM%i_MAP'%i,
                                         keys=setup.detwin+setup.insmode,
                                         which='best', required=1);
                bmaps.extend(tmp);

            mrx.compute_preproc (gp[0:argopt.max_file], bkg, bmaps, output=output);
            
        except Exception as exc:
            log.error ('Cannot compute PREPROC: '+str(exc));
            if argopt.debug == 'TRUE': raise;
        finally:
            log.closeFile ();

#
# Compute SPEC_CAL
#

if argopt.speccal != 'FALSE':

    # Read all products
    hdrs_calib = mrx.headers.loaddir (argopt.outputDir);

    # Group all PREPROC
    gps = mrx.headers.group (hdrs_calib, 'PREPROC', delta=argopt.delta, Delta=argopt.Delta,
                             keys=setup.detwin+setup.insmode+setup.fringewin,
                             continuous=argopt.cont);
    overwrite = (argopt.speccal == 'OVERWRITE');

    # Compute 
    for i,gp in enumerate(gps):
        try:
            log.info ('Compute SPEC_CAL {0} over {1} '.format(i+1,len(gps)));
            
            output = mrx.files.output (argopt.outputDir, gp[0], 'speccal');
            if os.path.exists (output+'.fits') and overwrite is False:
                log.info ('Product already exists');
                continue;

            log.setFile (output+'.log');
            
            mrx.compute_speccal (gp[0:argopt.max_file], output=output);
            
        except Exception as exc:
            log.error ('Cannot compute SPEC_CAL: '+str(exc));
            if argopt.debug == 'TRUE': raise;
        finally:
            log.closeFile ();

#
# Compute RTS
#

if argopt.rts != 'FALSE':

    # Read all calibration products
    hdrs_calib = mrx.headers.loaddir (argopt.outputDir);

    # Group all DATA
    gps = mrx.headers.group (hdrs_calib, 'DATA_PREPROC', delta=0,
                             keys=setup.detwin+setup.detmode+setup.insmode+setup.fringewin);
    overwrite = (argopt.rts == 'OVERWRITE');

    # Compute 
    for i,gp in enumerate(gps):
        try:
            log.info ('Compute RTS {0} over {1} '.format(i+1,len(gps)));
            
            output = mrx.files.output (argopt.outputDir, gp[0], 'rts');
            if os.path.exists (output+'.fits') and overwrite is False:
                log.info ('Product already exists');
                continue;

            log.setFile (output+'.log');

            speccal = mrx.headers.assoc (gp[0], hdrs_calib, 'SPEC_CAL',
                                         keys=setup.detwin+setup.insmode+setup.fringewin,
                                         which='best', required=1);

            # Associate MAP (best in this setup)
            bmaps = [];
            for i in range(1,7):
                tmp = mrx.headers.assoc (gp[0], hdrs_calib, 'BEAM%i_MAP'%i,
                                         keys=setup.detwin+setup.detmode+setup.insmode,
                                         which='best', required=1);
                bmaps.extend (tmp);
            
            # Associate KAPPA (closest in time)
            kappas = [];
            for i in range(1,7):
                keys = setup.detwin+setup.detmode+setup.insmode;
                tmp = mrx.headers.assoc (gp[0], hdrs_calib, 'BEAM%i_MAP'%i,
                                         keys=keys, which='closest', required=1);
                kappas.extend (tmp);
                
            mrx.compute_rts (gp, bmaps, kappas, speccal, output=output);

        except Exception as exc:
            log.error ('Cannot compute RTS: '+str(exc));
            if argopt.debug == 'TRUE': raise;
        finally:
            log.closeFile ();
            
#
# Compute VIS
#

if argopt.vis != 'FALSE':

    # Read all calibration products
    hdrs_calib = mrx.headers.loaddir (argopt.outputDir);

    # Group all DATA
    gps = mrx.headers.group (hdrs_calib, 'RTS', delta=0,
                             keys=setup.detwin+setup.detmode+setup.insmode+setup.fringewin);
    overwrite = (argopt.vis == 'OVERWRITE');

    # Compute 
    for i,gp in enumerate(gps):
        try:
            log.info ('Compute VIS {0} over {1} '.format(i+1,len(gps)));
            
            for nc in [1]:
                output = mrx.files.output (argopt.outputDir, gp[0], 'vis')+'_c%04i'%int(nc*10);
                if os.path.exists (output+'.fits') and overwrite is False:
                    log.info ('Product already exists');
                    continue;

                log.setFile (output+'.log');
                mrx.compute_vis (gp, output=output, ncoher=nc, threshold=2.0);

        except Exception as exc:
            log.error ('Cannot compute VIS: '+str(exc));
            if argopt.debug == 'TRUE': raise;
        finally:
            log.closeFile ();
            
                                    
#
# Compute VIS_CALIBRATED
#

if argopt.viscalib != 'FALSE':
    overwrite = (argopt.viscalib == 'OVERWRITE');

    # Read all calibration products, keep only the VIS
    hdrs = mrx.headers.loaddir (argopt.outputDir);

    # Group all VIS by calibratable setup
    keys = setup.detwin+setup.insmode+setup.fringewin+setup.visparam;
    gps = mrx.headers.group (hdrs, 'VIS', delta=1e9, Delta=1e9, keys=keys, continuous=False);

    # Parse input catalog
    catalog = mrx.headers.parse_argopt_catalog (argopt.calibrators);

    # Compute 
    for i,gp in enumerate (gps):
        try:
            log.info ('Calibrate setup {0} over {1} '.format(i+1,len(gps)));            
            log.setFile (argopt.outputDir+'/calibration_setup%i.log'%i);
            
            mrx.compute_all_viscalib (gp, catalog, outputDir=argopt.outputDir);

        except Exception as exc:
            log.error ('Cannot calibrate setup: '+str(exc));
            if argopt.debug == 'TRUE': raise;
        finally:
            log.closeFile ();


