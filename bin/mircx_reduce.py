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
  Run the mircx test pipeline. Shall be ran in the directory
  of the RAW data. The data format shall be .fits and/or .fits.fz
  The format .fits.gz is not supported.

  By default, outputs are written in directories preproc/ vis/
  and viscalib/
"""

epilog = \
"""
examples:
  cd /path/to/my/data/
  mirx_reduce.py
"""

parser = argparse.ArgumentParser (description=description, epilog=epilog,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter);

TrueFalse = ['TRUE','FALSE'];
TrueFalseOverwrite = ['TRUE','FALSE','OVERWRITE'];

parser.add_argument ("--debug", dest="debug",default='FALSE',
                     choices=TrueFalse,
                     help="stop or error");

parser.add_argument ("--max-file", dest="max_file",default=3000,type=int,
                     help="maximum number of file to load to build "
                          "product (speed-up for tests)");


parser.add_argument ("--raw-dir", dest="raw_dir",default='./',type=str,
                     help="directory of raw data");

parser.add_argument ("--preproc-dir", dest="preproc_dir",default='./preproc/',type=str,
                     help="directory of intermediate products");

parser.add_argument ("--rts-dir", dest="rts_dir",default='./rts/',type=str,
                     help="directory of intermediate products");

parser.add_argument ("--vis-dir", dest="vis_dir",default='./vis/',type=str,
                     help="directory of intermediate products");

parser.add_argument ("--vis-calibrated-dir", dest="viscalib_dir",default='./viscalib/',type=str,
                     help="directory of intermediate products");


parser.add_argument ("--background", dest="background",default='TRUE',
                     choices=TrueFalseOverwrite,
                     help="compute the BACKGROUND products");

parser.add_argument ("--beam-map", dest="bmap",default='TRUE',
                     choices=TrueFalseOverwrite,
                     help="compute the BEAM_MAP products");

parser.add_argument ("--preproc", dest="preproc",default='TRUE',
                     choices=TrueFalseOverwrite,
                     help="compute the PREPROC products");

parser.add_argument ("--spec-cal", dest="speccal",default='TRUE',
                     choices=TrueFalseOverwrite,
                     help="compute the SPEC_CAL products");

parser.add_argument ("--rts", dest="rts",default='TRUE',
                     choices=TrueFalseOverwrite,
                     help="compute the RTS products");

parser.add_argument ("--vis", dest="vis",default='TRUE',
                     choices=TrueFalseOverwrite,
                     help="compute the VIS products");

parser.add_argument ("--vis-calibrated", dest="viscalib",default='FALSE',
                     choices=TrueFalseOverwrite,
                     help="compute the VIS_CALIBRATED products");


parser.add_argument ("--calibrators", dest="calibrators",default='name1,diam,err,name2,diam,err',
                     type=str, help="list of calibration star with diameters");

parser.add_argument ("--snr-threshold", dest="snr_threshold", type=float,
                    default=2.0, help="SNR threshold for fringe selection");

parser.add_argument ("--ncoherent", dest="ncoherent", type=float,
                    default=2.0, help="number of frames (can be fractional) for coherent integration");

#
# Initialisation
#

# Parse argument
argopt = parser.parse_args ();

# Verbose
elog = log.trace ('mircx_reduce');

#
# Compute BACKGROUND_MEAN
#

if argopt.background != 'FALSE':
    overwrite = (argopt.background == 'OVERWRITE');

    # List inputs
    hdrs = hdrs_raw = mrx.headers.loaddir (argopt.raw_dir);
    
    # Group backgrounds
    keys = setup.detwin + setup.detmode + setup.insmode;
    gps = mrx.headers.group (hdrs, 'BACKGROUND', keys=keys,
                             delta=300, Delta=3600,
                             continuous=True);

    # Compute all backgrounds
    for i,gp in enumerate(gps):
        try:
            log.info ('Compute BACKGROUND_MEAN {0} over {1} '.format(i+1,len(gps)));
            
            output = mrx.files.output (argopt.preproc_dir, gp[0], 'bkg');
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
    overwrite = (argopt.bmap == 'OVERWRITE');
        
    # List inputs
    hdrs = mrx.headers.loaddir (argopt.raw_dir);
    hdrs_calib = mrx.headers.loaddir (argopt.preproc_dir);
    
    # Group all BEAMi
    keys = setup.detwin + setup.detmode + setup.insmode;
    gps = mrx.headers.group (hdrs, 'BEAM.*', keys=keys,
                             delta=300, Delta=3600,
                             continuous=True);

    # Compute all 
    for i,gp in enumerate(gps):
        try:
            log.info ('Compute BEAM_MAP {0} over {1} '.format(i+1,len(gps)));

            name = gp[0]['FILETYPE'].lower()+'map';
            output = mrx.files.output (argopt.preproc_dir, gp[0], name);
            if os.path.exists (output+'.fits') and overwrite is False:
                log.info ('Product already exists');
                continue;
            
            log.setFile (output+'.log');

            # Associate BACKGROUND
            keys = setup.detwin + setup.detmode + setup.insmode;
            bkg = mrx.headers.assoc (gp[0], hdrs_calib, 'BACKGROUND_MEAN',
                                     keys=keys, which='closest', required=1);
            
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
    overwrite = (argopt.preproc == 'OVERWRITE');

    # List inputs
    hdrs = mrx.headers.loaddir (argopt.raw_dir);
    hdrs_calib = mrx.headers.loaddir (argopt.preproc_dir);

    # Group all DATA
    keys = setup.detwin + setup.detmode + setup.insmode;
    gps = mrx.headers.group (hdrs, 'DATA', keys=keys,
                             delta=300, Delta=120,
                             continuous=True);

    # Compute 
    for i,gp in enumerate(gps):
        try:
            log.info ('Compute PREPROC {0} over {1} '.format(i+1,len(gps)));
            
            output = mrx.files.output (argopt.preproc_dir, gp[0], 'preproc');
            if os.path.exists (output+'.fits') and overwrite is False:
                log.info ('Product already exists');
                continue;

            log.setFile (output+'.log');

            # Associate BACKGROUND
            keys = setup.detwin + setup.detmode + setup.insmode;
            bkg  = mrx.headers.assoc (gp[0], hdrs_calib, 'BACKGROUND_MEAN',
                                     keys=keys, which='closest', required=1);

            # Associate BEAM_MAP (best of the night)
            bmaps = [];
            for i in range(1,7):
                keys = setup.detwin + setup.insmode;
                tmp = mrx.headers.assoc (gp[0], hdrs_calib, 'BEAM%i_MAP'%i,
                                         keys=keys, which='best', required=1);
                bmaps.extend (tmp);
            
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
    overwrite = (argopt.speccal == 'OVERWRITE');

    # List inputs
    hdrs = mrx.headers.loaddir (argopt.preproc_dir);

    # Group all PREPROC together
    keys = setup.detwin + setup.insmode + setup.fringewin;
    gps = mrx.headers.group (hdrs, 'DATA_PREPROC', keys=keys,
                             delta=1e20, Delta=1e20,
                             continuous=False);

    # Compute 
    for i,gp in enumerate(gps):
        try:
            log.info ('Compute SPEC_CAL {0} over {1} '.format(i+1,len(gps)));
            
            output = mrx.files.output (argopt.preproc_dir, gp[0], 'speccal');
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
    overwrite = (argopt.rts == 'OVERWRITE');

    # List inputs
    hdrs = mrx.headers.loaddir (argopt.preproc_dir);

    # Group all DATA
    keys = setup.detwin + setup.detmode + setup.insmode + setup.fringewin;
    gps = mrx.headers.group (hdrs, 'DATA_PREPROC', delta=0, keys=keys);

    # Compute 
    for i,gp in enumerate(gps):
        try:
            log.info ('Compute RTS {0} over {1} '.format(i+1,len(gps)));
            
            output = mrx.files.output (argopt.rts_dir, gp[0], 'rts');
            if os.path.exists (output+'.fits') and overwrite is False:
                log.info ('Product already exists');
                continue;

            log.setFile (output+'.log');

            # Associate SPEC_CAL
            keys = setup.detwin + setup.insmode + setup.fringewin;
            speccal = mrx.headers.assoc (gp[0], hdrs, 'SPEC_CAL', keys=keys,
                                         which='best', required=1);
            
            # Associate PROFILE (best BEAM_MAP in this setup)
            profiles = [];
            for i in range(1,7):
                keys = setup.detwin + setup.detmode + setup.insmode;
                tmp = mrx.headers.assoc (gp[0], hdrs, 'BEAM%i_MAP'%i,
                                         keys=keys, which='best', required=1);
                profiles.extend (tmp);

            # Associate KAPPA (closest BEAM_MAP in time, in this setup)
            kappas = [];
            for i in range(1,7):
                keys = setup.detwin + setup.detmode + setup.insmode;
                tmp = mrx.headers.assoc (gp[0], hdrs, 'BEAM%i_MAP'%i,
                                         keys=keys, which='closest', required=1);
                kappas.extend (tmp);
                
            mrx.compute_rts (gp, profiles, kappas, speccal, output=output);

        except Exception as exc:
            log.error ('Cannot compute RTS: '+str(exc));
            if argopt.debug == 'TRUE': raise;
        finally:
            log.closeFile ();
            
#
# Compute VIS
#

if argopt.vis != 'FALSE':
    overwrite = (argopt.vis == 'OVERWRITE');

    # List inputs
    hdrs = mrx.headers.loaddir (argopt.rts_dir);

    # Group all DATA
    keys = setup.detwin + setup.detmode + setup.insmode + setup.fringewin;
    gps = mrx.headers.group (hdrs, 'RTS', delta=0, keys=keys);

    # Compute 
    for i,gp in enumerate(gps):
        try:
            log.info ('Compute VIS {0} over {1} '.format(i+1,len(gps)));
            
            output = mrx.files.output (argopt.vis_dir, gp[0], 'vis');
            if os.path.exists (output+'.fits') and overwrite is False:
                log.info ('Product already exists');
                continue;

            log.setFile (output+'.log');
            mrx.compute_vis (gp, output=output, ncoher=argopt.ncoherent,
                             threshold=argopt.snr_threshold);

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
    hdrs = mrx.headers.loaddir (argopt.vis_dir);

    # Group all VIS by calibratable setup
    keys = setup.detwin + setup.insmode + setup.fringewin + setup.visparam;
    gps = mrx.headers.group (hdrs, 'VIS', delta=1e9, Delta=1e9,
                             keys=keys, continuous=False);

    # Parse input catalog
    catalog = mrx.headers.parse_argopt_catalog (argopt.calibrators);

    # Compute 
    for i,gp in enumerate (gps):
        try:
            log.info ('Calibrate setup {0} over {1} '.format(i+1,len(gps)));

            # Create output directory and set log
            mrx.files.ensure_dir (argopt.viscalib_dir);
            log.setFile (argopt.viscalib_dir+'/calibration_setup%i.log'%i);
            
            outputSetup = 'calibration_setup%i'%i;
            mrx.compute_all_viscalib (gp, catalog, outputDir=argopt.viscalib_dir,
                                      outputSetup=outputSetup);

        except Exception as exc:
            log.error ('Cannot calibrate setup: '+str(exc));
            if argopt.debug == 'TRUE': raise;
        finally:
            log.closeFile ();
