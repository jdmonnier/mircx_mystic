#! /usr/bin/env python                                                          
# -*- coding: iso-8859-15 -*-                                                


import mircx_mystic as mrx
import argparse
import glob
import os

from mircx_mystic import log, setup;
import datetime as datetime
import tkinter as tk
from tkinter import filedialog

#
# Implement options
#

# Describe the script
description = \
"""
description:
  Will create a night catalog summary directory with helpful files needed for rest of
  the pipeline. Will recognize fits, fits.fz files but NOT fits.gz

  The input and output directories are relative to the
  current directory.

  if you leave blank, the default identifier is today's date and raw data directory chosen by 
  dialog pickfile, out output directory is local.
"""

epilog = \
"""

Examples:
  
fully-specified:
  mircx_mystic_nightcat.py --raw-dir=/path/to/raw/data/ --mrx_dir=/path/to/reduced/data/ -id=JDM2022Jan04

defaults:
  cd /path/where/I/want/my/reduced/data/
  mirc_mystic_nightcat.py


"""

parser = argparse.ArgumentParser (description=description, epilog=epilog,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 add_help=False);


TrueFalse = ['TRUE','FALSE'];
TrueFalseOverwrite = ['TRUE','FALSE','OVERWRITE'];

nightcat = parser.add_argument_group ('(1) nightcat',
                    '\nCreates a summary directory containting\n'
                    'nightly summary in editable ASCII format and header info in panda dataframes');

nightcat.add_argument ("--raw-dir", dest="raw_dir",default=None,type=str,
                     help="directory of raw data [%(default)s]");

nightcat.add_argument ("--mrx-dir", dest="mirx_dir",default='./',type=str,
                     help="directory of mrx pipeline products [%(default)s]");

nightcat.add_argument ("--id", dest="mrx_id",
                     default=datetime.date.today().strftime('%Y%b%d'),type=str,
                     help="unique identifier for data reduction [%(default)s]");

advanced = parser.add_argument_group ('advanced user arguments');
                                        
advanced.add_argument ("--debug", dest="debug",default='FALSE',
                     choices=TrueFalse,
                     help="stop on error [%(default)s]");

advanced.add_argument ('--help', action='help',
                     help=argparse.SUPPRESS);

advanced.add_argument ('-h', action='help',
                     help=argparse.SUPPRESS);



#
# Initialization
#

# Parse argument
argopt = parser.parse_args ();

# Verbose
elog = log.trace ('mircx_mystic_nightcat');

# Set debug
if argopt.debug == 'TRUE':
    log.info ('start debug mode')
    import pdb;

#
# Compute NIGHT CATALOG and summary files, including header stuff.
#

if True == True:
    log.info("Starting Nightcat Routines:")

    #get raw directory if none passes
    if argopt.raw_dir == None:
        log.info("No Raw Diretory Passed. Using Dialog Pickfile")
        root = tk.Tk()
        root.withdraw()
        argopt.raw_dir = filedialog.askdirectory(title = "Select PATH to DATA")


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
            
            #if os.path.exists (output+'.fits') and overwrite is False:
            #    log.info ('Product already exists');
            #    continue;
                
            log.setFile (output+'.log');

            mrx.compute_background (gp[0:argopt.max_file],
                                    output=output, filetype=filetype, linear=argopt.linearize);
            
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
            
            #if os.path.exists (output+'.fits') and overwrite is False:
            #    log.info ('Product already exists');
            #    continue;
            
            log.setFile (output+'.log');

            # Associate BACKGROUND
            keys = setup.detwin + setup.detmode + setup.insmode;
            bkg = mrx.headers.assoc (gp[0], hdrs_calib, 'BACKGROUND_MEAN',
                                     keys=keys, which='closest', required=1);
            
            # Associate best FLAT based in gain
            flat = mrx.headers.assoc_flat (gp[0], hdrs_static);

            # Compute the BEAM_MAP
            mrx.compute_beam_map (gp[0:argopt.max_file], bkg, flat,argopt.threshold,
                                  output=output, filetype=filetype, linear=argopt.linearize);
            
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
            
            #if os.path.exists (output+'.fits') and overwrite is False:
            #    log.info ('Product already exists');
            #    continue;
            
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
            
            #if os.path.exists (output+'.fits') and overwrite is False:
            #    log.info ('Product already exists');
            #    continue;
            
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
            
            #if os.path.exists (output+'.fits') and overwrite is False:
            #    log.info ('Product already exists');
            #    continue;

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
                                 output=output, filetype=filetype, linear=argopt.linearize);
            
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
            
            #if os.path.exists (output+'.fits') and overwrite is False:
            #    log.info ('Product already exists');
            #    continue;

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
        

    ## Get rid of groups with low integration time
    nloads = [len(g) for g in gps]
    max_loads = max(nloads)
    gps = [g for g in gps if len(g)>(max_loads/2)]

    # Compute 
    for i,gp in enumerate(gps):
        try:
            log.info ('Compute OIFITS {0} over {1} '.format(i+1,len(gps)));

            if 'DATA' in gp[0]['FILETYPE']:
                filetype = 'OIFITS';
                data = True;
            else:
                filetype = gp[0]['FILETYPE'].replace('_RTS','_OIFITS');
                data = False
                
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
                             gdt_tincoh=argopt.gdt_tincoh,
                             ncs=argopt.ncs, nbs=argopt.nbs,
                             snr_threshold=argopt.snr_threshold if data else -1*argopt.snr_threshold, #catch this!
                             flux_threshold=argopt.flux_threshold, #keep this even foreground.
                             gd_attenuation=argopt.gd_attenuation if data else False,
                             gd_threshold=argopt.gd_threshold if data else 1e10, #keep all frames.
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
