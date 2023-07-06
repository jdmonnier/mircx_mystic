#! /usr/bin/env python
# -*- coding: iso-8859-15 -*-



#2023June. JDM Notes.
#I. Backgrounds. 
#  Most preproc steps depend on initial analysis of backgrounds.
# A. First load in all backgrounds, grouped by camera mode (loops reads frames/ramp gain), combiner, spectral mode, polarization (not HWP angle for now).
# B. within each group, measure the interference frequency and amplitude, identify bad pixels, find means and errors, estimate 
#   1. Analyse each ramp to detect contamination.
# Repeate for SKIES.
# 

# TODO 
#   1. organize groups based on detector and mode. loop over these groups
#   2. define background based on median filtering and determine noise. 
#   3. same for each shutter type.
#   4. then go through each file and attempt to check shutters.
#   5. alternatively just inspect each block in block file and mark badfiles .

import matplotlib.pyplot as plt
#import plotly.express as px
import mircx_mystic as mrx
import numpy as np
import argparse
import glob
import os
import sys
import pickle
import json

from mircx_mystic import log, setup, files, headers,preproc # JDM why pylance difference?
#from mircx_mystic import checkshutters
import datetime as datetime
import tkinter as tk
from tkinter import filedialog
import pandas as pd

#
# Implement options
#

# Describe the script
description = \
    """
description:
  Following nightcat, this will create preproc data.
  
  Only input is the summary directory, using keyword or dialog pickfile.

  if you leave blank, the SUMMARY directory chosen by 
  dialog pickfile, out output directory is local.
"""

epilog = \
    """

Examples:
  
fully-specified:
  mircx_mystic_preproc.py --summary-dir=/path/to/reduced/summary

defaults:
  cd /path/where/I/want/my/reduced/data/
  mirc_mystic_preproc.py


"""

parser = argparse.ArgumentParser(description=description, epilog=epilog,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 add_help=False)

TrueFalse = ['TRUE', 'FALSE']
TrueFalseOverwrite = ['TRUE', 'FALSE', 'OVERWRITE']

preproc_args = parser.add_argument_group('preproc arguments',
                                     '\nPreprocesses the raw data to create preproc data free of detector anolmalies.')

preproc_args.add_argument("--summary-dir", dest="summary_dir", default=None, type=str,
                      help="directory of SUMMARY  [%(default)s]")

preproc_args.add_argument("--log", dest="logLevel", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      default="INFO", help="log verbosity [%(default)s]")


advanced = parser.add_argument_group('advanced user arguments')

advanced.add_argument("--debug", dest="debug", default='FALSE',
                      choices=TrueFalse,
                      help="stop on error [%(default)s]")

advanced.add_argument('--help', action='help',
                      help=argparse.SUPPRESS)

advanced.add_argument('-h', action='help',
                      help=argparse.SUPPRESS)


#
# Initialization
#
str_to_remove = [' ','-','_','!','#','@','$','%','^','&','*','(',')']

# Parse argument
argopt = parser.parse_args()
log.setLevel(argopt.logLevel);

#
tempfile='.mircx_mystic_preproc.temp.log'
#remove tempfile if already exists
if os.path.exists(tempfile): os.remove(tempfile)
log.setFile(tempfile) ## will get renamed

# Verbose
elog = log.trace('mircx_mystic_preproc')  # for Timing.

# Set debug
if argopt.debug == 'TRUE':
    log.info('start debug mode')
    import pdb

#
#  Check Shutters... time consuming but important to do.
#
# get raw directory if none passes
if argopt.summary_dir == None:
    log.info("No Summary Directory Passed. Using Dialog Pickfile")
    root = tk.Tk()
    root.withdraw()
    argopt.summary_dir = filedialog.askdirectory(title="Select PATH to SUMMARY DIR",initialdir='./')

#Guard
if argopt.summary_dir[-8:] != '_SUMMARY':
    log.error("Choose valid SUMMARY directory. The following does not exist:\n %s "%(argopt.summary_dir))
    del elog
    log.closeFile()
    sys.exit()


# load json file to retrieve raw-dir, etc.
# USE the headers.csv file to create new block.csv (NO OVERWRITE)
# remake figures.
log.info("Chose SUMMARY directory %s"%(argopt.summary_dir))
log.info("     Will use saved headers,metadata,blocks")
mrx_root=argopt.summary_dir.split('/')[-1][:-8]
json_file=os.path.join(argopt.summary_dir,mrx_root+'_metadata.json')
with open(json_file) as f:
    jsonresult = json.load(f)
    f.close()
raw_dir=''
mrx_dir=''
mrx_root='' 
locals().update(jsonresult)
mrx_summary_dir=mrx_root+'_SUMMARY' # should match argopt.summary_dir
preproc_dir=mrx_root+'_PREPROC' # path to preproc.
# do not allow custom preproc. only ONE per instrument per night per SUMMARY directory.
path = os.path.join(mrx_dir, mrx_summary_dir)  # # should match argopt.summary_dir
preproc_path= os.path.join(path,preproc_dir)

if os.path.isfile(os.path.join(path,mrx_root+'_headers.csv')):
    phdrs=pd.read_csv(os.path.join(path,mrx_root+'_headers.csv'),low_memory=False)
else:
    log.critical("Could not find header file in %s"%(path));
    log.info("Try re-running mircx_mystic_nightcat again")
    del elog
    log.closeFile()
    sys.exit()
pblock_file=os.path.join(path,mrx_root+'_blocks.csv')
pblock = pd.read_csv(pblock_file,comment='#',low_memory=False)


hdrs=mrx.headers.p2h(phdrs)  # might change one day to use panda data frames throughout code...
blocks=mrx.headers.p2h(pblock)

hdrs=mrx.headers.updatehdrs(hdrs,blocks) # update headers with block info.

# Group backgrounds
#keys = setup.detwin + setup.detmode + setup.insmode+['OBJECT','MIRC STEPPER HWP_ELEVATOR POS','MIRC HWP0 POS'] # etalon? # same as blcoks?
keys = ['BLOCK']
gps = mrx.headers.group (hdrs, 'BACKGROUND', keys=keys,delta=1e20, Delta=1e20,continuous=True);
#gps = mrx.headers.group (hdrs, 'DATA', keys=keys,delta=1e20, Delta=1e20,continuous=True);

for gp in gps: 
    #print("HEADERS: LEN ",len(g),'\t', g[0]["OBJECT"],'\t',g[0]['CONF_NA'],'\t',g[0]['FILETYPE'],'\t',g[0]['FILENUM'],'-',g[-1]['FILENUM'] )
    b=[b for b in blocks if b['BLOCK'] == gp[0]['BLOCK'] ]
    #print("BLOCK:   LEN ",len(b),'\t',b[0]["OBJECT"],'\t',b[0]['CONFIG'],'\t',b[0]['FILETYPE'],'\t',b[0]['START'],'-',b[0]['END'] )
    
    # Compute all background ramp and statistics for each block in preproc directory
    filetype = 'BACKGROUND_MEAN' # goes in header
    output =  mrx.files.blockoutput(preproc_path, gp[0]['BLOCK'], 'bkg') # goes in filename
    log.info('Compute BACKGROUND_MEAN for BLOCK %i:' % (gp[0]['BLOCK']))
    log.info('Writing File: %s' % (output+'.fits'))

    # skip if file already exists
    if os.path.exists (output+'.fits') and overwrite is False:
        log.info ('%s already exists. Skipping creation.' % (output+'.fits'))
        continue;
    
    # will create a 
    #gp=gp[0:4]
    mrx.compute_background(gp, output=output, filetype=filetype) # this routine will do a lot!

#log.info('Cleanup memory')
del gps


breakpoint();


#TODO
#Fix phdrs based on blocks and meta data.
#  update ORIGNAME based on raw_dir. 
#  update FILETYPE based on BLOCK (actually all columns!)
#  Remove rows that aren't in block or in BADFILES 
#  make this into a funciton call.
#plt.show(block=False)

#temp={'bgarrays':bgarrays,'bgkeys':bgkeys}
#with open('temp_bgarray.pkl','wb') as f:
#    pickle.dump(temp,f)

with open('temp_bgarray.pkl','rb') as f:
    loaded_dict = pickle.load(f)
locals().update(loaded_dict)
#bgarrays,bgkeys = checkshutters.bgkeys(phdrs)

allprofiles,profilekeys = checkshutters.shutterprofiles (phdrs,bgarrays,bgkeys)

#temp={'allprofiles':allprofiles,'profilekeys':profilekeys}
#with open('temp_allarray.pkl','wb') as f:
#   pickle.dump(temp,f)

with open('temp_allarray.pkl','rb') as f:
    loaded_dict = pickle.load(f)
locals().update(loaded_dict)
plt.clf()
keylist=list(allprofiles.keys())
for key in keylist:
    plt.plot(allprofiles[key],label=key)
plt.legend()
plt.show()


#allarrays,allkeys=checkshutters.allshutterkeys(phdrs)
allk=list(allarrays.keys())
bgk=[allk[i][1:5] for i in range(len(allk))]
allprofiles={}
for allk0, bgk0 in zip(allk, bgk):
    temp = np.median( allarrays[allk0]-bgarrays[bgk0],axis=0)
    allprofiles[allk0]=temp # what to do about mystic background?

# did this work?


# Group backgrounds for each (gain, conf_na)
bg_phdrs = phdrs.loc[phdrs['FILETYPE'] =='BACKGROUND'] # select only Background
bg_hdrs= mrx.headers.p2h(bg_phdrs)
#bgfiles_gps=bg_phdrs.groupby(by=keys)['ORIGNAME'].apply(list)
#for bgfiles in bgfiles_gps:
#    for file in bgfiles:

keys = ['CONF_NA','GAIN','NLOOPS','NREADS']
bg_pgps = bg_phdrs.groupby(by=keys)
bg_dict = bg_pgps.indices
keylist=list(bg_dict.keys())
bgarrays={}
for key in keylist: # loop over all the key groups found. 
    print(key)
    print(bg_dict[key])
    tuple_keys=['NAXIS4','NAXIS3','NAXIS2','NAXIS1']
    #dimx,dimy=bg_hdrs[bg_dict[key][0]]['NAXIS1'] , bg_hdrs[bg_dict[key][0]]['NAXIS2']
    #DIMX=bg_hdrs[bg_dict[key][0]]['NAXIS2']
    nramps,nframes,dimx,dimy=[bg_hdrs[bg_dict[key][0]][temp0] for temp0 in tuple_keys] 
    bgtemp = np.zeros([dimx,dimy,len(bg_dict[key])])
    gaintest=np.zeros(len(bg_dict[key]))
    for i,file in enumerate(bg_dict[key]): 
        hdr0=[bg_hdrs[file]] # pass a list of 1 to next code.

        
        hdrcheck,cube,__ = files.load_raw (hdr0, coaddRamp='mean',
                            removeBias=False,differentiate=False,
                            saturationThreshold=None,
                            continuityThreshold=None,
                            linear=False,badpix=None,flat=None);
        nframes=hdrcheck['NAXIS3']
        nbin=hdrcheck['NBIN'] #JDM not debugged.
        if nframes < 4:
            breakpoint # will fail if frames per reset <4
        bgtemp[:,:,i] = (cube[0,-2,:,:]-cube[0,1,:,:])/(nframes-3.)/nbin
        gaintest[i]=hdrcheck['NAXIS3']
        #plt.plot(cube[0,:,10,20])
        #plt.clf()

        print(file)
    bgtemp.shape
    plt.clf()
    plt.plot(bgtemp[10,100,:])
    plt.plot(bgtemp[30,280,:])
    #plt.plot(cube[0,:,10,20])
    medbg = np.median(bgtemp,axis=2)
    bgarrays[key] = medbg
    #ig=px.imshow(bgtemp[:,:,0]-medbg)
    #fig.show()
    print('finish plt')


print("Check bgarry_list and keys")


#plt.clf()
#differentiate=True,
#              removeBias=True, background=None, coaddRamp=False,
#              badpix=None, flat=None, output='output',
#              saturationThreshold=60000,
#              continuityThreshold=10000,
#              linear=True): # depricate `linear` after testing


#keys = ['CONF_NA','GAIN']
#bg_pgps = bg_phdrs.groupby(by=keys)

#log.info(bg_pgps.size())

#ngroups = pgps.ngroups
#bg_dict = bg_pgps.indices
#keylist=list(bg_dict.keys())





#bg_gps = mrx.headers.group (hdrs, '.*', keys=keys,delta=1e20, Delta=1e20,continuous=True);

#for g in gps: 
#    print(g[0]["OBJECT"],'\t',g[0]['CONF_NA'],'\t',g[0]['FILETYPE'],'\t',g[0]['FILENUM'],'-',g[-1]['FILENUM'] )

group_first = [item[0] for item in gps]
group_last = [item[-1] for item in gps]

columns=['BLOCK','OBJECT','CONF_NA','HWP','FILETYPE','START','END']
block_dict= {}
block_dict['BLOCK']=list(range(len(group_first)))
block_dict['OBJECT']=[temp['OBJECT'] for temp in group_first]
block_dict['CONF_NA']=[temp['CONF_NA'] for temp in group_first]
block_dict['FILETYPE']=[temp['FILETYPE'] for temp in group_first]
block_dict['START']=[temp['FILENUM'] for temp in group_first]
block_dict['END']=[temp['FILENUM'] for temp in group_last]
pblock = pd.DataFrame(block_dict,columns=columns)
pblock_file=os.path.join(path,mrx_root+'_blocks.csv')
if os.path.exists(pblock_file): ## guard
    log.warning(pblock_file+' already exists. NOT OVERWRITING!! ')
    log.info('Loading old Block file')
    pblock = pd.read_csv(pblock_file,low_memory=False)
else:
    pblock.to_csv(pblock_file,index=False,sep=',')
    log.info("Writing block file:"+mrx_root+'_blocks.csv')


#blockdata=
#[temp['FILENUM'] for temp in group_first]
#blockdata = {'BLOCK': range(len(gps2)), 'OBJECT': 

#Students = {'Student': ['Amit', 'Cody',
#                        'Darren', 'Drew'],
#            'RollNumber': [1, 5, 10, 15],
#            'Grade': ['A', 'C', 'F', 'B']}
#df = pd.DataFrame(students,
##                  columns =['Student', 'RollNumber',
#                            'Grade'])
# displaying the original DataFrame
##Print("Original DataFrame")
#print(df)
 
# saving as a CSV file
#df.to_csv('Students.csv', sep ='\t')

del elog
log.closeFile()

# Move tempfile.
logfile =  os.path.join(path,mrx_root+'_preproc.log') 
fout=open(logfile,'a')
fin =open(tempfile,'r')
fout.writelines( fin.readlines() )
fout.close()
fin.close()
os.remove(tempfile)


exit()






# 
#hdrs_here[0].keys()  
#       phdrs=pd.DataFrame(hdrs_here)
        # pd.to_csv("file")
        # JDM for reasons I don't understand, I can't write good pickle/feather files
        # without 'cleaningup' through a csv file.  The speed to read/write csv is 
        # small, so will live with this but strange.

#for x in phdrs.keys(): print(x,phdrs[x].dtype)        
# List static calibrations. Don't use log of header since
# these static files can be updated by git regularly
hdrs_static = mrx.headers.loaddir(setup.static, uselog=False)

#
# Compute BACKGROUND_MEAN
#

# Group backgrounds
keys = setup.detwin + setup.detmode + setup.insmode + 'RESTART0'
gps = mrx.headers.group(hdrs, 'BACKGROUND', keys=keys,
                        delta=300, Delta=3600,
                        continuous=True)

# Compute all backgrounds
for i, gp in enumerate(gps):
    try:
        log.info(
            'Compute BACKGROUND_MEAN {0} over {1} '.format(i+1, len(gps)))

        filetype = 'BACKGROUND_MEAN'
        output = mrx.files.output(argopt.preproc_dir, gp[0], 'bkg')

        # if os.path.exists (output+'.fits') and overwrite is False:
        #    log.info ('Product already exists');
        #    continue;

        
        log.setFile(output+'.log')

        mrx.compute_background(gp[0:argopt.max_file],
                                output=output, filetype=filetype, linear=argopt.linearize)

    except Exception as exc:
        log.error('Cannot compute BACKGROUND_MEAN: '+str(exc))
        if argopt.debug == 'TRUE':
            pdb.post_mortem()
            raise
    finally:
        log.closeFile()

log.info('Cleanup memory')
del gps

#
# Compute BEAMi_MAP
#

# List inputs
hdrs_calib = mrx.headers.loaddir(argopt.preproc_dir)

# Group all BEAMi
keys = setup.detwin + setup.detmode + setup.insmode
gps = mrx.headers.group(hdrs, 'BEAM.*', keys=keys,
                        delta=300, Delta=3600,
                        continuous=True)

# Compute all
for i, gp in enumerate(gps):
    try:
        log.info('Compute BEAM_MAP {0} over {1} '.format(i+1, len(gps)))

        filetype = 'BEAM%i_MAP' % mrx.headers.get_beam(gp[0])
        output = mrx.files.output(argopt.preproc_dir, gp[0], filetype)

        # if os.path.exists (output+'.fits') and overwrite is False:
        #    log.info ('Product already exists');
        #    continue;

        log.setFile(output+'.log')

        # Associate BACKGROUND
        keys = setup.detwin + setup.detmode + setup.insmode
        bkg = mrx.headers.assoc(gp[0], hdrs_calib, 'BACKGROUND_MEAN',
                                keys=keys, which='closest', required=1)

        # Associate best FLAT based in gain
        flat = mrx.headers.assoc_flat(gp[0], hdrs_static)

        # Compute the BEAM_MAP
        mrx.compute_beam_map(gp[0:argopt.max_file], bkg, flat, argopt.threshold,
                                output=output, filetype=filetype, linear=argopt.linearize)

    except Exception as exc:
        log.error('Cannot compute BEAM_MAP: '+str(exc))
        if argopt.debug == 'TRUE':
            pdb.post_mortem()
            raise
    finally:
        log.closeFile()

log.info('Cleanup memory')
del hdrs_calib, gps

#
# Compute BEAM_MEAN
#

# List inputs
hdrs_calib = mrx.headers.loaddir(argopt.preproc_dir)

# Group all DATA
keys = setup.detwin + setup.insmode
gps = mrx.headers.group(hdrs_calib, 'BEAM._MAP', keys=keys,
                        delta=1e20, Delta=1e20,
                        continuous=False)

# Compute all
for i, gp in enumerate(gps):
    try:
        log.info('Compute BEAM_MEAN {0} over {1} '.format(i+1, len(gps)))

        filetype = 'BEAM%i_MEAN' % mrx.headers.get_beam(gp[0])
        output = mrx.files.output(argopt.preproc_dir, gp[0], filetype)

        # if os.path.exists (output+'.fits') and overwrite is False:
        #    log.info ('Product already exists');
        #    continue;

        log.setFile(output+'.log')

        # Compute the BEAM_MAP
        mrx.compute_beam_profile(gp[0:argopt.max_file],
                                    output=output, filetype=filetype)

    except Exception as exc:
        log.error('Cannot compute BEAM_MEAN: '+str(exc))
        if argopt.debug == 'TRUE':
            pdb.post_mortem()
            raise
    finally:
        log.closeFile()

log.info('Cleanup memory')
del gps

#
# Compute BEAM_PROFILE
#

# Group all DATA
keys = setup.detwin + setup.detmode + setup.insmode
gps = mrx.headers.group(hdrs_calib, 'BEAM._MAP', keys=keys,
                        delta=1e20, Delta=1e20,
                        continuous=False)

# Compute all
for i, gp in enumerate(gps):
    try:
        log.info(
            'Compute BEAM_PROFILE {0} over {1} '.format(i+1, len(gps)))

        filetype = 'BEAM%i_PROFILE' % mrx.headers.get_beam(gp[0])
        output = mrx.files.output(argopt.preproc_dir, gp[0], filetype)

        # if os.path.exists (output+'.fits') and overwrite is False:
        #    log.info ('Product already exists');
        #    continue;

        log.setFile(output+'.log')

        # Compute the BEAM_PROFILE
        mrx.compute_beam_profile(gp[0:argopt.max_file],
                                    output=output, filetype=filetype)

    except Exception as exc:
        log.error('Cannot compute BEAM_PROFILE: '+str(exc))
        if argopt.debug == 'TRUE':
            pdb.post_mortem()
            raise
    finally:
        log.closeFile()

log.info('Cleanup memory')
del hdrs_calib, gps

#
# Compute PREPROC
#

# List inputs
hdrs_calib = mrx.headers.loaddir(argopt.preproc_dir)

# Group all DATA
keys = setup.detwin + setup.detmode + setup.insmode
gps = mrx.headers.group(hdrs, 'DATA', keys=keys,
                        delta=120, Delta=argopt.max_integration_time_preproc,
                        continuous=True)

# Also reduce the FOREGROUND and BACKGROUND
if argopt.bbias != 'FALSE':
    gps += mrx.headers.group(hdrs, 'FOREGROUND', keys=keys,
                                delta=120, Delta=argopt.max_integration_time_preproc,
                                continuous=True)

    gps += mrx.headers.group(hdrs, 'BACKGROUND', keys=keys,
                                delta=120, Delta=argopt.max_integration_time_preproc,
                                continuous=True)

# Compute
for i, gp in enumerate(gps):
    try:
        log.info('Compute PREPROC {0} over {1} '.format(i+1, len(gps)))

        filetype = gp[0]['FILETYPE']+'_PREPROC'
        output = mrx.files.output(argopt.preproc_dir, gp[0], filetype)

        # if os.path.exists (output+'.fits') and overwrite is False:
        #    log.info ('Product already exists');
        #    continue;

        log.setFile(output+'.log')

        # Associate BACKGROUND
        keys = setup.detwin + setup.detmode + setup.insmode
        bkg = mrx.headers.assoc(gp[0], hdrs_calib, 'BACKGROUND_MEAN',
                                keys=keys, which='closest', required=1)

        # Associate best FLAT based in gain
        flat = mrx.headers.assoc_flat(gp[0], hdrs_static)

        # Associate BEAM_MEAN (best of the night)
        bmaps = []
        for i in range(1, 7):
            keys = setup.detwin + setup.insmode
            tmp = mrx.headers.assoc(gp[0], hdrs_calib, 'BEAM%i_MEAN' % i,
                                    keys=keys, which='best', required=1,
                                    quality=argopt.mean_quality)
            bmaps.extend(tmp)

        # Compute PREPROC
        mrx.compute_preproc(gp[0:argopt.max_file], bkg, flat, bmaps, argopt.threshold,
                            output=output, filetype=filetype, linear=argopt.linearize)

    except Exception as exc:
        log.error('Cannot compute PREPROC: '+str(exc))
        if argopt.debug == 'TRUE':
            pdb.post_mortem()
            raise
    finally:
        log.closeFile()

log.info('Cleanup memory')
del hdrs, hdrs_calib, gps

#
# Compute SPEC_CAL
#

# List inputs
hdrs = mrx.headers.loaddir(argopt.preproc_dir)

# Group all PREPROC together
keys = setup.detwin + setup.insmode + setup.fringewin
gps = mrx.headers.group(hdrs, 'DATA_PREPROC', keys=keys,
                        delta=1e20, Delta=1e20,
                        continuous=False)

# Compute
for i, gp in enumerate(gps):
    try:
        log.info('Compute SPEC_CAL {0} over {1} '.format(i+1, len(gps)))

        filetype = 'SPEC_CAL'
        output = mrx.files.output(argopt.preproc_dir, gp[0], filetype)

        # if os.path.exists (output+'.fits') and overwrite is False:
        #    log.info ('Product already exists');
        #    continue;

        log.setFile(output+'.log')

        mrx.compute_speccal(gp[0:argopt.max_file],
                            output=output, filetype=filetype,
                            fitorder=argopt.speccal_order)

    except Exception as exc:
        log.error('Cannot compute SPEC_CAL: '+str(exc))
        if argopt.debug == 'TRUE':
            pdb.post_mortem()
            raise
    finally:
        log.closeFile()

log.info('Cleanup memory')
del hdrs, gps


#
# Compute RTS
#

if argopt.rts != 'FALSE':
    overwrite = (argopt.rts == 'OVERWRITE')

# List inputs
hdrs = mrx.headers.loaddir(argopt.preproc_dir)

# Reduce DATA
keys = setup.detwin + setup.detmode + setup.insmode + setup.fringewin
gps = mrx.headers.group(hdrs, 'DATA_PREPROC', delta=120,
                        Delta=argopt.max_integration_time_preproc, keys=keys)

# Reduce FOREGROUND and BACKGROUND
if argopt.bbias != 'FALSE':
    gps += mrx.headers.group(hdrs, 'FOREGROUND_PREPROC', delta=120,
                                Delta=argopt.max_integration_time_preproc, keys=keys)

    gps += mrx.headers.group(hdrs, 'BACKGROUND_PREPROC', keys=keys,
                                delta=120, Delta=argopt.max_integration_time_preproc,
                                continuous=True)
# Compute
for i, gp in enumerate(gps):
    try:
        log.info('Compute RTS {0} over {1} '.format(i+1, len(gps)))

        filetype = gp[0]['FILETYPE'].replace('_PREPROC', '_RTS')
        output = mrx.files.output(argopt.rts_dir, gp[0], filetype)

        if os.path.exists(output+'.fits') and overwrite is False:
            log.info('Product already exists')
            continue

        log.setFile(output+'.log')

        # Associate SPEC_CAL
        keys = setup.detwin + setup.insmode + setup.fringewin
        speccal = mrx.headers.assoc(gp[0], hdrs, 'SPEC_CAL', keys=keys,
                                    which='best', required=1, quality=0.001)

        # Associate PROFILE
        profiles = []
        for i in range(1, 7):
            keys = setup.detwin + setup.detmode + setup.insmode
            tmp = mrx.headers.assoc(gp[0], hdrs, 'BEAM%i_PROFILE' % i,
                                    keys=keys, which='best', required=1,
                                    quality=argopt.profile_quality)
            profiles.extend(tmp)

        # Associate KAPPA (closest BEAM_MAP in time, in this setup,
        # with sufficient quality)
        kappas = []
        for i in range(1, 7):
            keys = setup.detwin + setup.detmode + setup.insmode
            if argopt.kappa_gain == 'FALSE':
                keys.remove('GAIN')
            tmp = mrx.headers.assoc(gp[0], hdrs, 'BEAM%i_MAP' % i,
                                    keys=keys, quality=argopt.kappa_quality,
                                    which='closest', required=1)
            kappas.extend(tmp)

        mrx.compute_rts(gp, profiles, kappas, speccal,
                        output=output, filetype=filetype,
                        save_all_freqs=argopt.save_all_freqs)

        # If remove PREPROC
        if argopt.rm_preproc == 'TRUE':
            for g in gp:
                f = g['ORIGNAME']
                log.info('Remove the PREPROC: '+f)
                os.remove(f)

    except Exception as exc:
        log.error('Cannot compute RTS: '+str(exc))
        if argopt.debug == 'TRUE':
            pdb.post_mortem()
            raise
    finally:
        log.closeFile()

log.info('Cleanup memory')
del hdrs, gps


#
# Compute BBIAS_COEFF
#

if argopt.bbias != 'FALSE':
    overwrite = (argopt.bbias == 'OVERWRITE')

    # List inputs
    hdrs = mrx.headers.loaddir(argopt.rts_dir)

    # Group all DATA_RTS
    keys = setup.detwin + setup.detmode + setup.insmode + setup.fringewin
    #keys = setup.target_names
    gps = mrx.headers.group(hdrs, 'DATA_RTS', keys=keys,
                            delta=1e20, Delta=1e20,
                            continuous=False)

    # Compute
    for i, gp in enumerate(gps):
        try:
            log.info('Compute BBIAS_COEFF {0} over {1} '.format(i+1, len(gps)))

            filetype = 'BBIAS_COEFF'
            output = mrx.files.output(argopt.rts_dir, gp[0], filetype)

            if os.path.exists(output+'.fits') and overwrite is False:
                log.info('Product already exists')
                continue

            log.setFile(output+'.log')

            # Associate BACKGROUND_RTS
            keys = setup.detwin + setup.detmode + setup.insmode
            #keys = setup.target_names
            bkg = mrx.headers.assoc(gp[0], hdrs, 'BACKGROUND_RTS',
                                    keys=keys, which='all', required=1)

            # Associate FOREGROUND_RTS
            keys = setup.detwin + setup.detmode + setup.insmode
            #keys = setup.target_names
            fg = mrx.headers.assoc(gp[0], hdrs, 'FOREGROUND_RTS',
                                   keys=keys, which='all', required=1)

            # Making the computation
            mrx.compute_bbias_coeff(gp, bkg, fg, argopt.ncoherent, output=output,
                                    filetype=filetype)

        except Exception as exc:
            log.error('Cannot compute '+filetype+': '+str(exc))
            if argopt.debug == 'TRUE':
                pdb.post_mortem()
                raise
        finally:
            log.closeFile()

    log.info('Cleanup memory')
    del hdrs, gps

#
# Compute the trends
#

if argopt.selection != 'FALSE':
    overwrite = (argopt.selection == 'OVERWRITE')

    # List inputs
    hdrs = mrx.headers.loaddir(argopt.rts_dir)

    # Group all DATA by night
    gps = mrx.headers.group(hdrs, 'DATA_RTS', delta=1e9,
                            Delta=1e9, keys=[], continuous=False)

    # Only one output for the entire directory
    output = mrx.files.output(argopt.oifits_dir, 'night', 'selection')

    if os.path.exists(output+'.fits') and overwrite is False:
        log.info('Product already exists')

    else:
        try:
            log.setFile(output+'.log')

            # Compute
            mrx.compute_selection(gps[0], output=output, filetype='SELECTION',
                                  interactive=False, ncoher=10, nscan=64)

        except Exception as exc:
            log.error('Cannot compute SELECTION: '+str(exc))
            if argopt.debug == 'TRUE':
                pdb.post_mortem()
                raise
        finally:
            log.closeFile()


#
# Compute OIFITS
#

if argopt.oifits != 'FALSE':
    overwrite = (argopt.oifits == 'OVERWRITE')

    # List inputs
    hdrs = mrx.headers.loaddir(argopt.rts_dir)

    # Group all DATA
    keys = setup.detwin + setup.detmode + setup.insmode + setup.fringewin
    gps = mrx.headers.group(hdrs, 'DATA_RTS', delta=120,
                            Delta=argopt.max_integration_time_oifits, keys=keys)

    # Include FOREGROUND and BACKGROUND
    if argopt.reduce_foreground == 'TRUE':
        gps += mrx.headers.group(hdrs, 'FOREGROUND_RTS', delta=120,
                                 Delta=argopt.max_integration_time_oifits, keys=keys)

        gps += mrx.headers.group(hdrs, 'BACKGROUND_RTS', keys=keys,
                                 delta=120, Delta=argopt.max_integration_time_oifits,
                                 continuous=True)

    # Get rid of groups with low integration time
    nloads = [len(g) for g in gps]
    max_loads = max(nloads)
    gps = [g for g in gps if len(g) > (max_loads/2)]

    # Compute
    for i, gp in enumerate(gps):
        try:
            log.info('Compute OIFITS {0} over {1} '.format(i+1, len(gps)))

            if 'DATA' in gp[0]['FILETYPE']:
                filetype = 'OIFITS'
                data = True
            else:
                filetype = gp[0]['FILETYPE'].replace('_RTS', '_OIFITS')
                data = False

            output = mrx.files.output(argopt.oifits_dir, gp[0], filetype)

            if os.path.exists(output+'.fits') and overwrite is False:
                log.info('Product already exists')
                continue

            log.setFile(output+'.log')

            # Associate BBIAS_COEFF
            keys = setup.detwin + setup.detmode + setup.insmode
            if argopt.bbias == 'FALSE':
                coeff = []
            else:
                coeff = mrx.headers.assoc(gp[0], hdrs, 'BBIAS_COEFF',
                                          keys=keys, which='best', required=0)

            mrx.compute_vis(gp, coeff, output=output,
                            filetype=filetype,
                            ncoher=argopt.ncoherent,
                            gdt_tincoh=argopt.gdt_tincoh,
                            ncs=argopt.ncs, nbs=argopt.nbs,
                            snr_threshold=argopt.snr_threshold if data else -
                            1*argopt.snr_threshold,  # catch this!
                            # keep this even foreground.
                            flux_threshold=argopt.flux_threshold,
                            gd_attenuation=argopt.gd_attenuation if data else False,
                            # keep all frames.
                            gd_threshold=argopt.gd_threshold if data else 1e10,
                            vis_reference=argopt.vis_reference)

            # If remove RTS
            if argopt.rm_rts == 'TRUE':
                for g in gp:
                    f = g['ORIGNAME']
                    log.info('Remove the RTS: '+f)
                    os.remove(f)

        except Exception as exc:
            log.error('Cannot compute OIFITS: '+str(exc))
            if argopt.debug == 'TRUE':
                pdb.post_mortem()
                raise
        finally:
            log.closeFile()

    log.info('Cleanup memory')
    del hdrs, gps


# Delete elog to have final
# print of execution time
del elog



