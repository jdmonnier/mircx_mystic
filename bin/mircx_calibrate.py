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
  Shall be ran in the directory of the OIFITS data.
  The data format shall be .fits and/or .fits.fz
  The format .fits.gz is not supported.

  By default, outputs are written in directory ./calibrated/
"""

epilog = \
"""
examples:
  cd /path/to/my/data/oifits/
  mirx_calibrate.py --calibrators=HD1234,0.75,0.1
"""

parser = argparse.ArgumentParser (description=description, epilog=epilog,
                                 formatter_class=argparse.RawDescriptionHelpFormatter);

TrueFalse = ['TRUE','FALSE'];
TrueOverwrite = ['TRUE','OVERWRITE'];
TrueFalseOverwrite = ['TRUE','FALSE','OVERWRITE'];

parser.add_argument ("--debug", dest="debug",default='FALSE',
                     choices=TrueFalse,
                     help="stop or error");

parser.add_argument ("--max-file", dest="max_file",default=3000,type=int,
                     help="maximum number of file to load to build "
                          "product (speed-up for tests)");


parser.add_argument ("--oifits-dir", dest="oifits_dir",default='./',type=str,
                     help="directory of input visibilities [%(default)s]");

parser.add_argument ("--oifits-calibrated-dir", dest="oifitscalib_dir",default='./calibrated/',type=str,
                     help="directory of output calibrated visibilities [%(default)s]");


parser.add_argument ("--oifits-calibrated", dest="oifitscalib",default='TRUE',
                     choices=TrueOverwrite,
                     help="compute the OIFITS_CALIBRATED products [%(default)s]");


parser.add_argument ("--calibrators", dest="calibrators",default='name1,diam,err,name2,diam,err',
                     type=str, help="list of calibration star with diameters and error "
                                    "in mas and in the form name1,diam,err,name2,diam,err...");

parser.add_argument ("--delta-tf", dest="delta_tf",default=0.05,
                     type=float, help="interpolation time in days [%(default)s]");

parser.add_argument ("--lbd-min", dest="lbd_min",default=1.5,
                     type=float, help="minimum wavelenght in um [%(default)s]");

parser.add_argument ("--lbd-max", dest="lbd_max",default=1.72,
                     type=float, help="maximum wavelenght in um [%(default)s]");

parser.add_argument ("--flag-edges", dest="flag_edges",default='TRUE',
                    choices=TrueFalse,
                    help="flag first and last channels [%(default)s]");

parser.add_argument ("--use-detmode", dest="use_detmode",default='TRUE',
                    choices=TrueFalse,
                    help="use detector parameters to associate calibrators [%(default)s]");

#
# Initialisation
#

# Parse argument
argopt = parser.parse_args ();

# Verbose
elog = log.trace ('mircx_calibrate');
                                    
#
# Compute OIFITS_CALIBRATED
#

if argopt.oifitscalib != 'FALSE':
    overwrite = (argopt.oifitscalib == 'OVERWRITE');

    # Read all calibration products, keep only the OIFITS
    hdrs = mrx.headers.loaddir (argopt.oifits_dir);

    # Define the calibratable setups
    keys = setup.detwin + setup.insmode + \
           setup.fringewin + setup.visparam + setup.beamorder + setup.pop;
           
    if argopt.use_detmode == 'TRUE': keys += setup.detmode;

    # Group all OIFITS by calibratable setup
    gps = mrx.headers.group (hdrs, 'OIFITS', delta=1e9, Delta=1e9,
                             keys=keys, continuous=False);

    # Parse input catalog
    catalog = mrx.headers.parse_argopt_catalog (argopt.calibrators);
    
    # Update missing information by on-line query
    # mrx.headers.update_diam_from_jmmc (catalog);

    # Compute 
    for i,gp in enumerate (gps):
        try:
            log.info ('Calibrate setup {0} over {1} '.format(i+1,len(gps)));

            # Create output directory and set log
            mrx.files.ensure_dir (argopt.oifitscalib_dir);
            log.setFile (argopt.oifitscalib_dir+'/calibration_setup%i.log'%i);
            
            outputSetup = 'calibration_setup%i'%i;
            mrx.compute_all_viscalib (gp, catalog, outputDir=argopt.oifitscalib_dir,
                                      outputSetup=outputSetup,
                                      deltaTf=argopt.delta_tf,
                                      lbdMin=argopt.lbd_min,
                                      lbdMax=argopt.lbd_max,
                                      flagEdges=argopt.flag_edges,
                                      keys=keys);

        except Exception as exc:
            log.error ('Cannot calibrate setup: '+str(exc));
            if argopt.debug == 'TRUE': raise;
        finally:
            log.closeFile ();
