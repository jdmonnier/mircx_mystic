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
  Run the mircx pipeline. The RAW data format shall
  be .fits and/or .fits.fz  The format .fits.gz
  is not supported.

  The reduction is decomposed into 3 steps: preproc, rts,
  oifits. Each have them can be (des)activated, or tuned,
  with the following options.

  The input and output directories are relative to the
  current directory.

"""

epilog = \
"""
examples:

  # Run the entire reduction
  
  cd /path/where/I/want/my/reduced/data/
  mircx_reduce.py --raw-dir=/path/to/raw/data/

  # Do the preproc step only
  
  mircx_reduce.py --raw-dir=/path/to/raw/data/ --rts=FALSE --oifits=FALSE

  # Rerun the oifits step only, use a different
  # threshold for SNR selection, dump the results
  # into a different directory
  
  mircx_reduce.py --preproc=FALSE --rts=FALSE snr-threshold=4.0 --oifits-dir=oifits_new


"""

parser = argparse.ArgumentParser (description=description, epilog=epilog,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 add_help=False);


TrueFalse = ['TRUE','FALSE'];
TrueFalseOverwrite = ['TRUE','FALSE','OVERWRITE'];

preproc = parser.add_argument_group ('(1) preproc',
                    '\nCreates  the BACKGROUND, BEAM, PREPROC and SPEC_CAL\n'
                    'intermediate products, which mostly corresponds to\n'
                    'detector images cleaned from the detector artifact.');

preproc.add_argument ("--preproc", dest="preproc",default='TRUE',
                     choices=TrueFalseOverwrite,
                     help="compute the PREPROC products [%(default)s]");

preproc.add_argument ("--raw-dir", dest="raw_dir",default='./',type=str,
                     help="directory of raw data [%(default)s]");

preproc.add_argument ("--preproc-dir", dest="preproc_dir",default='./preproc/',type=str,
                     help="directory of products [%(default)s]");

preproc.add_argument ("--max-integration-time-preproc", dest="max_integration_time_preproc",
                      default=30.,type=float,
                      help='maximum integration into a single file, in (s).\n'
                      'This apply to PREPROC, and RTS steps [%(default)s]');

preproc.add_argument ("--threshold", dest="threshold",default=5.,type=float,
                     help='threshold in sigma for identifying bad pixels [%(default)s]');

preproc.add_argument ("--mean-quality", dest="mean_quality", type=float,
                     default=20.0, help="minimum quality to consider the BEAM_MEAN as a valid window for cropping the data in the preproc [%(default)s]");

preproc.add_argument ("--speccal-order", dest="speccal_order", type=int,
                    default=2, help="order of polynomial fitting in spectral calib [%(default)s]");

rts = parser.add_argument_group ('(2) rts',
                  '\nCreates RTS intermediate products, which are the\n'
                  'coherent flux and the photometric flux in real time,\n'
                  'cleaned from the instrumental behavior.');

rts.add_argument ("--rts", dest="rts",default='TRUE',
                  choices=TrueFalseOverwrite,
                  help="compute the RTS products [%(default)s]");

rts.add_argument ("--rts-dir", dest="rts_dir",default='./rts/',type=str,
                  help="directory of products [%(default)s]");

rts.add_argument ("--profile-quality", dest="profile_quality", type=float,
                  default=20.0, help="minimum quality to consider the BEAM_PROFILE as a valid profile for extraction [%(default)s]");

rts.add_argument ("--kappa-quality", dest="kappa_quality", type=float,
                  default=5.0, help="minimum quality to consider the BEAM_MAP as a valid kappa matrix [%(default)s]");

rts.add_argument ("--kappa-gain", dest="kappa_gain",default='TRUE',
                  choices=TrueFalse,
                  help="use GAIN to associate kappa [%(default)s]");

rts.add_argument ("--save-all-freqs", dest="save_all_freqs",default='FALSE',
                  choices=TrueFalse,
                  help="save the entire FFTs in RTS file [%(default)s]");

rts.add_argument ("--rm-preproc", dest="rm_preproc",default='FALSE',
                  choices=TrueFalse,
                  help="rm the PREPROC file after computing the RTS [%(default)s]");

oifits = parser.add_argument_group ('(3) oifits',
                     '\nCreates the final OIFITS products, which are the\n' 
                     'uncalibrated mean visibilities and closure phases\n'
                     'computed by selecting and averaging the data in RTS.');

oifits.add_argument ("--oifits", dest="oifits",default='TRUE',
                     choices=TrueFalseOverwrite,
                     help="compute the OIFITS products [%(default)s]");

oifits.add_argument ("--oifits-dir", dest="oifits_dir",default='./oifits/',type=str,
                     help="directory of products [%(default)s]");

oifits.add_argument ("--max-integration-time-oifits", dest="max_integration_time_oifits",
                      default=150.,type=float,
                      help='maximum integration into a single file, in (s).\n'
                      'This apply to OIFITS steps [%(default)s]');

oifits.add_argument ("--ncoherent", dest="ncoherent", type=int,
                     default=5, help="number of frames for coherent integration [%(default)s]");

oifits.add_argument ("--nincoherent", dest="nincoherent", type=int,
                     default=5, help="number of ramps for incoherent integration [%(default)s]");

oifits.add_argument ("--ncs", dest="ncs", type=int,
                     default=1, help="number of frame-offset for cross-spectrum [%(default)s]");

oifits.add_argument ("--nbs", dest="nbs", type=int,
                     default=4, help='only used when bbias=FALSE, this is the shift for the bispectrum [%(default)s]');

oifits.add_argument ("--snr-threshold", dest="snr_threshold", type=float,
                     default=2.0, help="SNR threshold for fringe rejection [%(default)s]");

oifits.add_argument ("--flux-threshold", dest="flux_threshold", type=float,
                     default=20.0, help="FLUX threshold for rejection [%(default)s]");

oifits.add_argument ("--gd-threshold", dest="gd_threshold", type=float,
                     default=30.0, help="GD threshold for rejection in um [%(default)s]");

oifits.add_argument ("--gd-attenuation", dest="gd_attenuation",default='TRUE',
                     choices=TrueFalse,
                     help="correct from the attenuation due to GD [%(default)s]");

oifits.add_argument ("--vis-reference", dest="vis_reference",default='self',
                     choices=['self','spec-diff'],
                     help="phase reference for VIS estimator [%(default)s]");

oifits.add_argument ("--rm-rts", dest="rm_rts",default='FALSE',
                     choices=TrueFalse,
                     help="rm the RTS file after computing the OIFITS [%(default)s]");


advanced = parser.add_argument_group ('advanced user arguments');
                                         
advanced.add_argument ("--bbias", dest="bbias",default='TRUE',
                     choices=TrueFalseOverwrite,
                     help="compute the BBIAS_COEFF product [%(default)s]");

advanced.add_argument ("--reduce-foreground", dest="reduce_foreground",default='FALSE',
                     choices=TrueFalse,
                     help="reduce the FOREGROUND into OIFITS [%(default)s]");

advanced.add_argument ("--debug", dest="debug",default='FALSE',
                     choices=TrueFalse,
                     help="stop on error [%(default)s]");

advanced.add_argument ("--max-file", dest="max_file",default=3000,type=int,
                     help=argparse.SUPPRESS);

advanced.add_argument ('--help', action='help',
                     help=argparse.SUPPRESS);

advanced.add_argument ('-h', action='help',
                     help=argparse.SUPPRESS);

advanced.add_argument ('--selection', dest="selection",default='FALSE',
                     choices=TrueFalseOverwrite,
                    help=argparse.SUPPRESS);



#
# Initialisation
#

# Parse argument
argopt = parser.parse_args ();

# Verbose
elog = log.trace ('mircx_reduce');

# Set debug
if argopt.debug == 'TRUE':
    log.info ('start debug mode')
    import pdb;

## force choices when --bbias=TRUE
if argopt.bbias != 'FALSE':
    log.info ('bbias is TRUE so force save-all-freqs=TRUE');
    argopt.save_all_freqs = 'TRUE';
    argopt.ncs = 1;
    argopt.nbs = 0;

    
#
# Compute Preproc
#

if argopt.preproc != 'FALSE':
    overwrite = (argopt.preproc == 'OVERWRITE');

    # List inputs
    hdrs = mrx.headers.loaddir (argopt.raw_dir);

    # List static calibrations. Don't use log of header since
    # these static files can be updated by git regularly
    hdrs_static = mrx.headers.loaddir (setup.static, uselog=False);

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

            filetype = 'BACKGROUND_MEAN';
            output = mrx.files.output (argopt.preproc_dir, gp[0], 'bkg');
            
            if os.path.exists (output+'.fits') and overwrite is False:
                log.info ('Product already exists');
                continue;
                
            log.setFile (output+'.log');

            mrx.compute_background (gp[0:argopt.max_file],
                                    output=output, filetype=filetype);
            
        except Exception as exc:
            log.error ('Cannot compute BACKGROUND_MEAN: '+str(exc));
            if argopt.debug == 'TRUE': pdb.post_mortem(); raise; 
        finally:
            log.closeFile ();

    log.info ('Cleanup memory');
    del gps;
    
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
            
            filetype = 'BEAM%i_MAP'%mrx.headers.get_beam (gp[0]);
            output = mrx.files.output (argopt.preproc_dir, gp[0], filetype);
            
            if os.path.exists (output+'.fits') and overwrite is False:
                log.info ('Product already exists');
                continue;
            
            log.setFile (output+'.log');

            # Associate BACKGROUND
            keys = setup.detwin + setup.detmode + setup.insmode;
            bkg = mrx.headers.assoc (gp[0], hdrs_calib, 'BACKGROUND_MEAN',
                                     keys=keys, which='closest', required=1);
            
            # Associate best FLAT based in gain
            flat = mrx.headers.assoc_flat (gp[0], hdrs_static);

            # Compute the BEAM_MAP
            mrx.compute_beam_map (gp[0:argopt.max_file], bkg, flat,argopt.threshold,
                                  output=output, filetype=filetype);
            
        except Exception as exc:
            log.error ('Cannot compute BEAM_MAP: '+str(exc));
            if argopt.debug == 'TRUE':  pdb.post_mortem(); raise;
        finally:
            log.closeFile ();
        
    log.info ('Cleanup memory');
    del hdrs_calib, gps;
    
    #
    # Compute BEAM_MEAN
    #
    
    # List inputs
    hdrs_calib = mrx.headers.loaddir (argopt.preproc_dir);
    
    # Group all DATA
    keys = setup.detwin + setup.insmode;
    gps = mrx.headers.group (hdrs_calib, 'BEAM._MAP', keys=keys,
                             delta=1e20, Delta=1e20,
                             continuous=False);

    # Compute all
    for i,gp in enumerate(gps):
        try:
            log.info ('Compute BEAM_MEAN {0} over {1} '.format(i+1,len(gps)));

            filetype = 'BEAM%i_MEAN'%mrx.headers.get_beam (gp[0]);
            output = mrx.files.output (argopt.preproc_dir, gp[0], filetype);
            
            if os.path.exists (output+'.fits') and overwrite is False:
                log.info ('Product already exists');
                continue;
            
            log.setFile (output+'.log');

            # Compute the BEAM_MAP
            mrx.compute_beam_profile (gp[0:argopt.max_file],
                                      output=output, filetype=filetype);
            
        except Exception as exc:
            log.error ('Cannot compute BEAM_MEAN: '+str(exc));
            if argopt.debug == 'TRUE': pdb.post_mortem(); raise; 
        finally:
            log.closeFile ();
        
    log.info ('Cleanup memory');
    del gps;

    #
    # Compute BEAM_PROFILE
    #
    
    # Group all DATA
    keys = setup.detwin + setup.detmode + setup.insmode;
    gps = mrx.headers.group (hdrs_calib, 'BEAM._MAP', keys=keys,
                             delta=1e20, Delta=1e20,
                             continuous=False);

    # Compute all 
    for i,gp in enumerate(gps):
        try:
            log.info ('Compute BEAM_PROFILE {0} over {1} '.format(i+1,len(gps)));

            filetype = 'BEAM%i_PROFILE'%mrx.headers.get_beam (gp[0]);
            output = mrx.files.output (argopt.preproc_dir, gp[0], filetype);
            
            if os.path.exists (output+'.fits') and overwrite is False:
                log.info ('Product already exists');
                continue;
            
            log.setFile (output+'.log');

            # Compute the BEAM_PROFILE
            mrx.compute_beam_profile (gp[0:argopt.max_file],
                                      output=output, filetype=filetype);
            
        except Exception as exc:
            log.error ('Cannot compute BEAM_PROFILE: '+str(exc));
            if argopt.debug == 'TRUE': pdb.post_mortem(); raise; 
        finally:
            log.closeFile ();
        
    log.info ('Cleanup memory');
    del hdrs_calib, gps;

    #
    # Compute PREPROC
    #

    # List inputs
    hdrs_calib = mrx.headers.loaddir (argopt.preproc_dir);
    

    # Group all DATA
    keys = setup.detwin + setup.detmode + setup.insmode;
    gps = mrx.headers.group (hdrs, 'DATA', keys=keys,
                             delta=120, Delta=argopt.max_integration_time_preproc,
                             continuous=True);

    # Also reduce the FOREGROUND and BACKGROUND
    if argopt.bbias != 'FALSE':
        gps += mrx.headers.group (hdrs, 'FOREGROUND', keys=keys,
                                  delta=120, Delta=argopt.max_integration_time_preproc,
                                  continuous=True);
        
        gps += mrx.headers.group (hdrs, 'BACKGROUND', keys=keys,
                                  delta=120, Delta=argopt.max_integration_time_preproc,
                                  continuous=True);

    # Compute 
    for i,gp in enumerate(gps):
        try:
            log.info ('Compute PREPROC {0} over {1} '.format(i+1,len(gps)));

            filetype = gp[0]['FILETYPE']+'_PREPROC';
            output = mrx.files.output (argopt.preproc_dir, gp[0], filetype);
            
            if os.path.exists (output+'.fits') and overwrite is False:
                log.info ('Product already exists');
                continue;

            log.setFile (output+'.log');

            # Associate BACKGROUND
            keys = setup.detwin + setup.detmode + setup.insmode;
            bkg  = mrx.headers.assoc (gp[0], hdrs_calib, 'BACKGROUND_MEAN',
                                     keys=keys, which='closest', required=1);

            # Associate best FLAT based in gain
            flat = mrx.headers.assoc_flat (gp[0], hdrs_static);
                
            # Associate BEAM_MEAN (best of the night)
            bmaps = [];
            for i in range(1,7):
                keys = setup.detwin + setup.insmode;
                tmp = mrx.headers.assoc (gp[0], hdrs_calib, 'BEAM%i_MEAN'%i,
                                         keys=keys, which='best', required=1,
                                         quality=argopt.mean_quality);
                bmaps.extend (tmp);

            # Compute PREPROC
            mrx.compute_preproc (gp[0:argopt.max_file], bkg, flat, bmaps,argopt.threshold,
                                 output=output, filetype=filetype);
            
        except Exception as exc:
            log.error ('Cannot compute PREPROC: '+str(exc));
            if argopt.debug == 'TRUE': pdb.post_mortem(); raise; 
        finally:
            log.closeFile ();

    log.info ('Cleanup memory');
    del hdrs, hdrs_calib, gps;
    
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
            
            filetype = 'SPEC_CAL';
            output = mrx.files.output (argopt.preproc_dir, gp[0], filetype);
            
            if os.path.exists (output+'.fits') and overwrite is False:
                log.info ('Product already exists');
                continue;

            log.setFile (output+'.log');
            
            mrx.compute_speccal (gp[0:argopt.max_file],
                                 output=output, filetype=filetype,
                                 fitorder=argopt.speccal_order);
            
        except Exception as exc:
            log.error ('Cannot compute SPEC_CAL: '+str(exc));
            if argopt.debug == 'TRUE': pdb.post_mortem(); raise; 
        finally:
            log.closeFile ();

    log.info ('Cleanup memory');
    del hdrs, gps;

    

#
# Compute RTS
#

if argopt.rts != 'FALSE':
    overwrite = (argopt.rts == 'OVERWRITE');

    # List inputs
    hdrs = mrx.headers.loaddir (argopt.preproc_dir);

    # Reduce DATA
    keys = setup.detwin + setup.detmode + setup.insmode + setup.fringewin;
    gps = mrx.headers.group (hdrs, 'DATA_PREPROC', delta=120,
                             Delta=argopt.max_integration_time_preproc, keys=keys);

    # Reduce FOREGROUND and BACKGROUND
    if argopt.bbias != 'FALSE':
        gps += mrx.headers.group (hdrs, 'FOREGROUND_PREPROC', delta=120,
                                  Delta=argopt.max_integration_time_preproc, keys=keys);
        
        gps += mrx.headers.group (hdrs, 'BACKGROUND_PREPROC', keys=keys,
                                  delta=120, Delta=argopt.max_integration_time_preproc,
                                  continuous=True);
    # Compute 
    for i,gp in enumerate(gps):
        try:
            log.info ('Compute RTS {0} over {1} '.format(i+1,len(gps)));

            filetype = gp[0]['FILETYPE'].replace('_PREPROC','_RTS');
            output = mrx.files.output (argopt.rts_dir, gp[0], filetype);
            
            if os.path.exists (output+'.fits') and overwrite is False:
                log.info ('Product already exists');
                continue;

            log.setFile (output+'.log');

            # Associate SPEC_CAL
            keys = setup.detwin + setup.insmode + setup.fringewin;
            speccal = mrx.headers.assoc (gp[0], hdrs, 'SPEC_CAL', keys=keys,
                                         which='best', required=1, quality=0.001);
            
            # Associate PROFILE
            profiles = [];
            for i in range(1,7):
                keys = setup.detwin + setup.detmode + setup.insmode;
                tmp = mrx.headers.assoc (gp[0], hdrs, 'BEAM%i_PROFILE'%i,
                                         keys=keys, which='best', required=1,
                                         quality=argopt.profile_quality);
                profiles.extend (tmp);

            # Associate KAPPA (closest BEAM_MAP in time, in this setup,
            # with sufficient quality)
            kappas = [];
            for i in range(1,7):
                keys = setup.detwin + setup.detmode + setup.insmode;
                if argopt.kappa_gain == 'FALSE': keys.remove ('GAIN');
                tmp = mrx.headers.assoc (gp[0], hdrs, 'BEAM%i_MAP'%i,
                                         keys=keys, quality=argopt.kappa_quality,
                                         which='closest', required=1);
                kappas.extend (tmp);
                
            mrx.compute_rts (gp, profiles, kappas, speccal,
                             output=output, filetype=filetype,
                             save_all_freqs=argopt.save_all_freqs);

            # If remove PREPROC
            if argopt.rm_preproc == 'TRUE':
                for g in gp:
                    f = g['ORIGNAME'];
                    log.info ('Remove the PREPROC: '+f);
                    os.remove (f);

        except Exception as exc:
            log.error ('Cannot compute RTS: '+str(exc));
            if argopt.debug == 'TRUE': pdb.post_mortem(); raise;
        finally:
            log.closeFile ();
            
    log.info ('Cleanup memory');
    del hdrs, gps;
    

#
# Compute BBIAS_COEFF
#

if argopt.bbias != 'FALSE':
    overwrite = (argopt.bbias == 'OVERWRITE');

    # List inputs
    hdrs = mrx.headers.loaddir (argopt.rts_dir);

    # Group all DATA_RTS
    keys = setup.detwin + setup.detmode + setup.insmode + setup.fringewin;
    #keys = setup.target_names
    gps = mrx.headers.group (hdrs, 'DATA_RTS', keys=keys,
                             delta=1e20, Delta=1e20,
                             continuous=False);
    
    # Compute 
    for i,gp in enumerate(gps):
        try:
            log.info ('Compute BBIAS_COEFF {0} over {1} '.format(i+1,len(gps)));

            filetype = 'BBIAS_COEFF';
            output = mrx.files.output (argopt.rts_dir, gp[0], filetype);
            
            if os.path.exists (output+'.fits') and overwrite is False:
                log.info ('Product already exists');
                continue;

            log.setFile (output+'.log');

            # Associate BACKGROUND_RTS
            keys = setup.detwin + setup.detmode + setup.insmode;
            #keys = setup.target_names
            bkg = mrx.headers.assoc (gp[0], hdrs, 'BACKGROUND_RTS',
                                     keys=keys, which='all', required=1);

            # Associate FOREGROUND_RTS
            keys = setup.detwin + setup.detmode + setup.insmode;
            #keys = setup.target_names
            fg = mrx.headers.assoc (gp[0], hdrs, 'FOREGROUND_RTS',
                                   keys=keys, which='all', required=1);

            # Making the computation
            mrx.compute_bbias_coeff (gp, bkg, fg, argopt.ncoherent, output=output,
                                     filetype=filetype);

        except Exception as exc:
            log.error ('Cannot compute '+filetype+': '+str(exc));
            if argopt.debug == 'TRUE': pdb.post_mortem(); raise; 
        finally:
            log.closeFile ();
            
    log.info ('Cleanup memory');
    del hdrs, gps;

#
# Compute the trends
#

if argopt.selection != 'FALSE':
    overwrite = (argopt.selection == 'OVERWRITE');

    # List inputs
    hdrs = mrx.headers.loaddir (argopt.rts_dir);

    # Group all DATA by night
    gps = mrx.headers.group (hdrs, 'DATA_RTS', delta=1e9,
                             Delta=1e9, keys=[], continuous=False);

    # Only one output for the entire directory
    output = mrx.files.output (argopt.oifits_dir, 'night', 'selection');
    
    if os.path.exists (output+'.fits') and overwrite is False:
        log.info ('Product already exists');
        
    else:
        try:
            log.setFile (output+'.log');
    
            # Compute
            mrx.compute_selection (gps[0], output=output, filetype='SELECTION',
                                   interactive=False, ncoher=10, nscan=64);
        
        except Exception as exc:
            log.error ('Cannot compute SELECTION: '+str(exc));
            if argopt.debug == 'TRUE': pdb.post_mortem(); raise;
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
    gps = mrx.headers.group (hdrs, 'DATA_RTS', delta=120,
                             Delta=argopt.max_integration_time_oifits, keys=keys);

    # Include FOREGROUND and BACKGROUND
    if argopt.reduce_foreground == 'TRUE':
        gps += mrx.headers.group (hdrs, 'FOREGROUND_RTS', delta=120,
                                  Delta=argopt.max_integration_time_oifits, keys=keys);
        
        gps += mrx.headers.group (hdrs, 'BACKGROUND_RTS', keys=keys,
                                  delta=120, Delta=argopt.max_integration_time_oifits,
                                  continuous=True);

    # Compute 
    for i,gp in enumerate(gps):
        try:
            log.info ('Compute OIFITS {0} over {1} '.format(i+1,len(gps)));

            if 'DATA' in gp[0]['FILETYPE']:
                filetype = 'OIFITS';
            else:
                filetype = gp[0]['FILETYPE'].replace('_RTS','_OIFITS');
                
            output = mrx.files.output (argopt.oifits_dir, gp[0], filetype);
            
            if os.path.exists (output+'.fits') and overwrite is False:
                log.info ('Product already exists');
                continue;

            log.setFile (output+'.log');

            # Associate BBIAS_COEFF
            keys = setup.detwin + setup.detmode + setup.insmode;
            if argopt.bbias == 'FALSE':
                coeff = [];
            else:
                coeff = mrx.headers.assoc (gp[0], hdrs, 'BBIAS_COEFF',
                                        keys=keys, which='best', required=0);
            
            mrx.compute_vis (gp, coeff, output=output,
                             filetype=filetype,
                             ncoher=argopt.ncoherent,
                             nincoher=argopt.nincoherent,
                             ncs=argopt.ncs, nbs=argopt.nbs,
                             snr_threshold=argopt.snr_threshold,
                             flux_threshold=argopt.flux_threshold,
                             gdAttenuation=argopt.gd_attenuation,
                             gd_threshold=argopt.gd_threshold,
                             vis_reference=argopt.vis_reference);
            
            # If remove RTS
            if argopt.rm_rts == 'TRUE':
                for g in gp:
                    f = g['ORIGNAME'];
                    log.info ('Remove the RTS: '+f);
                    os.remove (f);

        except Exception as exc:
            log.error ('Cannot compute OIFITS: '+str(exc));
            if argopt.debug == 'TRUE': pdb.post_mortem(); raise;
        finally:
            log.closeFile ();
            
    log.info ('Cleanup memory');
    del hdrs, gps;


    
# Delete elog to have final
# pring of execution time
del elog;
