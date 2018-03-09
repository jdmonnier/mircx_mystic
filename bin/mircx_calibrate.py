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
  Run the mircx pipeline for calibration.
  Shall be ran in the directory of the VIS data.
  The data format shall be .fits and/or .fits.fz
  The format .fits.gz is not supported.

  By default, outputs are written in directory ./viscalib/
"""

epilog = \
"""
examples:
  cd /path/to/my/data/vis/
  mirx_calibrate.py --calibrators=HD1234,0.75,0.1
"""

parser = argparse.ArgumentParser (description=description, epilog=epilog,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter);

TrueFalse = ['TRUE','FALSE'];
TrueOverwrite = ['TRUE','OVERWRITE'];
TrueFalseOverwrite = ['TRUE','FALSE','OVERWRITE'];

parser.add_argument ("--debug", dest="debug",default='FALSE',
                     choices=TrueFalse,
                     help="stop or error");

parser.add_argument ("--max-file", dest="max_file",default=3000,type=int,
                     help="maximum number of file to load to build "
                          "product (speed-up for tests)");


parser.add_argument ("--vis-dir", dest="vis_dir",default='./',type=str,
                     help="directory of input visibilities");

parser.add_argument ("--vis-calibrated-dir", dest="viscalib_dir",default='./viscalib/',type=str,
                     help="directory of output calibrated visibilities");


parser.add_argument ("--vis-calibrated", dest="viscalib",default='TRUE',
                     choices=TrueOverwrite,
                     help="compute the VIS_CALIBRATED products");


parser.add_argument ("--calibrators", dest="calibrators",default='name1,diam,err,name2,diam,err',
                     type=str, help="list of calibration star with diameters");

parser.add_argument ("--delta-tf", dest="delta_tf",default=0.05,
                     type=float, help="interpolation time in [days]");

parser.add_argument ("--lbd-min", dest="lbd_min",default=1.5,
                     type=float, help="minimum wavelenght [um]");

parser.add_argument ("--lbd-max", dest="lbd_max",default=1.72,
                     type=float, help="maximum wavelenght [um]");

#
# Initialisation
#

# Parse argument
argopt = parser.parse_args ();

# Verbose
elog = log.trace ('mircx_calibrate');
                                    
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
                                      outputSetup=outputSetup,
                                      deltaTf=argopt.delta_tf,
                                      lbdMin=argopt.lbd_min,
                                      lbdMax=argopt.lbd_max);

        except Exception as exc:
            log.error ('Cannot calibrate setup: '+str(exc));
            if argopt.debug == 'TRUE': raise;
        finally:
            log.closeFile ();
