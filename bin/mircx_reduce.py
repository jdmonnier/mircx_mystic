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
  Run the mircx pipeline. Shall be ran in the directory
  of the RAW data. The data format shall be .fits and/or .fits.fz
  The format .fits.gz is not supported.

  By default, outputs are written in the following directories:

  preproc/ contains the BACKGROUND, BEAM, PREPROC and SPEC_CAL
           intermediate products, which mostly corresponds to
           detector images cleaned from the detector artifact
           (step 1 of reduction).

  rts/     contains RTS intermediate products, which are the
           coherent flux and the photometric flux in real time,
           cleaned from the instrumental behavior, but not yet
           averaged (step 2 of reduction).

  oifits/  contains the final OIFITS products, which are the 
           uncalibrated mean visibilities and closure phases
           computed by selecting and averaging the data in RTS
           (step 3 of reduction).

"""

epilog = \
"""
examples:
  cd /path/to/my/data/
  mirx_reduce.py
"""

parser = argparse.ArgumentParser (description=description, epilog=epilog,
                                 formatter_class=argparse.RawDescriptionHelpFormatter);


TrueFalse = ['TRUE','FALSE'];
TrueFalseOverwrite = ['TRUE','FALSE','OVERWRITE'];

parser.add_argument ("--debug", dest="debug",default='FALSE',
                     choices=TrueFalse,
                     help="stop or error");

parser.add_argument ("--max-file", dest="max_file",default=3000,type=int,
                     help="maximum number of file to load to build "
                          "product (speed-up for tests) [%(default)s]");


parser.add_argument ("--raw-dir", dest="raw_dir",default='./',type=str,
                     help="directory of raw data [%(default)s]");

parser.add_argument ("--preproc-dir", dest="preproc_dir",default='./preproc/',type=str,
                     help="directory of products [%(default)s]");

parser.add_argument ("--rts-dir", dest="rts_dir",default='./rts/',type=str,
                     help="directory of products [%(default)s]");

parser.add_argument ("--oifits-dir", dest="oifits_dir",default='./oifits/',type=str,
                     help="directory of products [%(default)s]");


parser.add_argument ("--preproc", dest="preproc",default='TRUE',
                     choices=TrueFalseOverwrite,
                     help="compute the PREPROC products [%(default)s]");

parser.add_argument ("--rts", dest="rts",default='TRUE',
                     choices=TrueFalseOverwrite,
                     help="compute the RTS products [%(default)s]");

parser.add_argument ("--oifits", dest="oifits",default='TRUE',
                     choices=TrueFalseOverwrite,
                     help="compute the OIFITS products [%(default)s]");


parser.add_argument ("--beam-quality", dest="beam_quality", type=float,
                    default=2.0, help="minimum quality to consider the beammap as valid [%(default)s]");

parser.add_argument ("--ncoherent", dest="ncoherent", type=float,
                    default=2.0, help="number of frames for coherent integration, can be fractional [%(default)s]");

parser.add_argument ("--snr-threshold", dest="snr_threshold", type=float,
                    default=2.0, help="SNR threshold for fringe selection [%(default)s]");

# Private arguments
parser.add_argument ("--kappa-gain", dest="kappa_gain",default='TRUE',
                     choices=TrueFalse,
                     help="Use GAIN to associate kappa");


#
# Initialisation
#

# Parse argument
argopt = parser.parse_args ();

# Verbose
elog = log.trace ('mircx_reduce');

# List inputs
hdrs = hdrs_raw = mrx.headers.loaddir (argopt.raw_dir);
    
if argopt.preproc != 'FALSE':
    overwrite = (argopt.preproc == 'OVERWRITE');

    #
    # Compute BACKGROUND_MEAN
    #

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

    # List inputs
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

    # List inputs
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

            # Associate KAPPA (closest BEAM_MAP in time, in this setup,
            # with sufficient quality)
            kappas = [];
            for i in range(1,7):
                keys = setup.detwin + setup.detmode + setup.insmode;
                if argopt.kappa_gain == 'FALSE': keys.remove ('GAIN');
                tmp = mrx.headers.assoc (gp[0], hdrs, 'BEAM%i_MAP'%i,
                                         keys=keys, quality=argopt.beam_quality,
                                         which='closest', required=1);
                kappas.extend (tmp);
                
            mrx.compute_rts (gp, profiles, kappas, speccal, output=output);

        except Exception as exc:
            log.error ('Cannot compute RTS: '+str(exc));
            if argopt.debug == 'TRUE': raise;
        finally:
            log.closeFile ();
            
#
# Compute OIFITS
#

if argopt.oifits != 'FALSE':
    overwrite = (argopt.oifits == 'OVERWRITE');

    # List inputs
    hdrs = mrx.headers.loaddir (argopt.rts_dir);

    # Group all DATA
    keys = setup.detwin + setup.detmode + setup.insmode + setup.fringewin;
    gps = mrx.headers.group (hdrs, 'RTS', delta=0, keys=keys);

    # Compute 
    for i,gp in enumerate(gps):
        try:
            log.info ('Compute OIFITS {0} over {1} '.format(i+1,len(gps)));
            
            output = mrx.files.output (argopt.oifits_dir, gp[0], 'oifits');
            if os.path.exists (output+'.fits') and overwrite is False:
                log.info ('Product already exists');
                continue;

            log.setFile (output+'.log');
            mrx.compute_vis (gp, output=output, ncoher=argopt.ncoherent,
                             threshold=argopt.snr_threshold);

        except Exception as exc:
            log.error ('Cannot compute OIFITS: '+str(exc));
            if argopt.debug == 'TRUE': raise;
        finally:
            log.closeFile ();
            
