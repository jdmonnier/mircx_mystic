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
"""

epilog = \
"""
examples:
  cd /path/to/my/data/
  mirx_preproc.py --background=FALSE
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

parser.add_argument ("--preproc-dir", dest="preproc_dir",default='./preproc/',type=str,
                     help="output directories");

parser.add_argument ("--rts-dir", dest="rts_dir",default='./rts/',type=str,
                     help="output directories");

parser.add_argument ("--vis-dir", dest="vis_dir",default='./vis/',type=str,
                     help="output directories");

parser.add_argument ("--vis-calibrated-dir", dest="viscalib_dir",default='./viscalib/',type=str,
                     help="output directories");


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

#
# Compute BACKGROUND_MEAN
#

if argopt.background != 'FALSE':
    overwrite = (argopt.background == 'OVERWRITE');

    # List inputs
    hdrs = hdrs_raw = mrx.headers.loaddir ('./');
    
    # Group backgrounds
    keys = setup.detwin+setup.detmode+setup.insmode;
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
    hdrs = mrx.headers.loaddir ('./');
    hdrs_calib = mrx.headers.loaddir (argopt.preproc_dir);
    
    # Group all BEAMi
    keys = setup.detwin+setup.detmode+setup.insmode;
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

            # Associate background
            keys = setup.detwin+setup.detmode+setup.insmode;
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
    hdrs = mrx.headers.loaddir ('./');
    hdrs_calib = mrx.headers.loaddir (argopt.preproc_dir);

    # Group all DATA
    keys = setup.detwin+setup.detmode+setup.insmode;
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

            # Associate background
            keys = setup.detwin+setup.detmode+setup.insmode;
            bkg  = mrx.headers.assoc (gp[0], hdrs_calib, 'BACKGROUND_MEAN',
                                     keys=keys, which='closest', required=1);

            # Associate beam map
            bmaps = [];
            for i in range(1,7):
                keys = setup.detwin+setup.insmode;
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
    keys = setup.detwin+setup.insmode+setup.fringewin;
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
    keys = setup.detwin+setup.detmode+setup.insmode+setup.fringewin;
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
            keys = setup.detwin+setup.insmode+setup.fringewin;
            speccal = mrx.headers.assoc (gp[0], hdrs, 'SPEC_CAL', keys=keys,
                                         which='best', required=1);
            
            # Associate BEAM_MAP
            bmaps = [];
            for i in range(1,7):
                keys = setup.detwin+setup.detmode+setup.insmode;
                tmp = mrx.headers.assoc (gp[0], hdrs, 'BEAM%i_MAP'%i,
                                         keys=keys, which='best', required=1);
                bmaps.extend (tmp);
            
            mrx.compute_rts (gp, bmaps, speccal, output=output);

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
    keys = setup.detwin+setup.detmode+setup.insmode+setup.fringewin;
    gps = mrx.headers.group (hdrs, 'RTS', delta=0, keys=keys);

    # Compute 
    for i,gp in enumerate(gps):
        try:
            log.info ('Compute VIS {0} over {1} '.format(i+1,len(gps)));
            
            for nc in [1]:
                output = mrx.files.output (argopt.vis_dir, gp[0], 'vis')+'_c%04i'%int(nc*10);
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
    hdrs = mrx.headers.loaddir (argopt.vis_dir);

    # Group all VIS by calibratable setup
    keys = setup.detwin+setup.insmode+setup.fringewin+setup.visparam;
    gps = mrx.headers.group (hdrs, 'VIS', delta=1e9, Delta=1e9,
                             keys=keys, continuous=False);

    # Parse input catalog
    catalog = mrx.headers.parse_argopt_catalog (argopt.calibrators);

    # Compute 
    for i,gp in enumerate (gps):
        try:
            log.info ('Calibrate setup {0} over {1} '.format(i+1,len(gps)));            
            log.setFile (argopt.viscal_dir+'/calibration_setup%i.log'%i);
            
            mrx.compute_all_viscalib (gp, catalog, outputDir=argopt.viscal_dir);

        except Exception as exc:
            log.error ('Cannot calibrate setup: '+str(exc));
            if argopt.debug == 'TRUE': raise;
        finally:
            log.closeFile ();