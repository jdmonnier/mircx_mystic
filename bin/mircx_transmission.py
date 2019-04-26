#! /usr/bin/env python                                                          
# -*- coding: iso-8859-15 -*-                                                   

import mircx_pipeline as mrx
import argparse, glob, os
from datetime import datetime
import numpy as np
from astropy.io import fits as pyfits
import matplotlib.pyplot as plt

from mircx_pipeline import log, headers, plot
from mircx_pipeline.headers import HMQ

# Describe the script
description = \
"""
description:
 Plot a report of the transmission across multiple
 nights of observations.

"""

epilog = \
"""
examples: 

  python mircx_transmission.py --num-nights=10

 or
  
  python mircx_transmission.py --date-from=2018Oct25 --date-to=2018Oct29

"""

TrueFalse = ['TRUE','FALSE']

parser = argparse.ArgumentParser(description=description, epilog=epilog,
          formatter_class=argparse.RawDescriptionHelpFormatter,
          add_help=True)

parser.add_argument("--dir", dest="dir",default='./',type=str,
                    help="main trunk of reduction directory [%(default)s]")
parser.add_argument("--num-nights",dest="num_of_nights",default=0,type=int,
                    help="Number of nights to be included in plot [50]")
parser.add_argument("--date-from",dest="night_from",default='',type=str,
                    help="Earliest date to be included in plot (YYYYMmmDD)")
parser.add_argument("--date-to",dest="night_to",default='',type=str,
                    help="Latest date to be included in plot (YYYYMmmDD)")
parser.add_argument("--targ-list",dest="targ_list",default='mircx_targets.list',type=str,
                    help="local database with SCI and CAL IDs [%(default)s]")
parser.add_argument("--only-reference", dest="only_reference",default='FALSE',
                    choices=TrueFalse,
                    help="Use only REFERENCE (calibrator) stars [%(default)s]")


# Parse arguments:
argopt = parser.parse_args()

# Verbose:
elog = log.trace('mircx_transm')

# Check how many nights are to be plotted:
now = datetime.now()
if argopt.num_of_nights != 0:
    nNight = argopt.num_of_nights
else:
    if argopt.night_from == '':
        nNight = 50 # default to plotting the 50 most recent nights of data
    else:
        fNight = argopt.night_from
        try:
            fN = datetime.strptime(fNight,'%Y%b%d')
        except ValueError:
            log.error('Argument "date-from" does not match format "%Y%b%d"')
            sys.exit()
        if argopt.night_to == '':
            lNight = now.strftime('%Y%b%d') # takes current date in YYYMmmDD format
        else:
            lNight = argopt.night_to

# Get the list of observation dates from directory trunk:
if argopt.dir == './':
    sDir = ''
else:
    sDir = argopt.dir
    if sDir[-1] != '/':
        sDir = sDir+'/'

dirList = glob.glob(sDir+'*') # read in directory names from directory trunk
dL = list(set([d.split('_')[0] for d in dirList])) # remove duplicate dates
for d in dL:
    try:
        dL1.append(datetime.strptime(d,'%Y%b%d')) # for sorting, translate these into datetime format
    except NameError:
        # first instance:
        dL1 = []
    except ValueError:
        # ensure other things in the directory are skipped over but keep a note of what they are
        log.info('Skipping files in current directory: '+d)

dL2 = [dL1[i] for i in np.argsort(dL1)] # sort the dates (earliest first)
dateList = [d.strftime('%Y%b%d') for d in dL2] # convert these back into their original format
try:
    if len(dateList) > nNight:
        log.info('Number of observation dates exceeds '+str(nNight))
        dL3 = dateList[len(dateList)-nNight:]
        dateList = dL3
        log.info('Cropped earlier dates from dateList')
    else:
        log.info('Number of observation dates is less than '+str(nNight))
        log.info('All observation nights in current directory will be plotted')
except NameError:
    # catch instances where fNight and lNight are used to limit date range rather than nNight
    dL3 = dateList[dateList.index(fNight):dateList.index(lNight)]
    dateList = dL3
    log.info('Removed dates earlier than '+fNight+' from dateList')
    if lNight != now.strftime('%Y%b%d'):
        log.info('Removed dates later than '+lNight+' from dateList')

# Locate calibrator names
if not os.path.isfile(os.environ['MIRCX_PIPELINE']+'/mircx_pipeline/'+argopt.targ_list):
    log.error(os.environ['MIRCX_PIPELINE']+'/mircx_pipeline/'+argopt.targ_list+' not found!')
    log.info('Please rectify this before continuing')
    sys.exit()
else:
    localDB = os.environ['MIRCX_PIPELINE']+'/mircx_pipeline/'+argopt.targ_list
    # ^-- this is the local target history database

calL = []
with open(localDB) as input:
    head = input.readline()
    for line in input:
        if line.split(',')[5] == 'CAL':
            calL.append(line.split(',')[0])

# Load astroquery
try:
    from astroquery.vizier import Vizier;
    log.info ('Load astroquery.vizier');
except:
    log.warning ('Cannot load astroquery.vizier, try:');
    log.warning ('sudo conda install -c astropy astroquery');

# ----------------------------
# Date non-specific values for calculating the transmission:
# ----------------------------
# Zero point of 2MASS:H from Cohen et al. (2003, AJ 126, 1090):
Hzp = 9.464537e6 # [photons/millisec/m2/mircons]
# internal transmission * quantum efficiency from Cyprien [dimensionless]:
iTQE    = 0.5
# collecting area of 1 telescope (assuming circular aperture) [m2]:
telArea = np.pi * 0.5*0.5


# ----------------------------
# Set up the plot window:
# ----------------------------
fig,axes = plt.subplots(7,1,sharex=True,figsize=(16,9))
plot.compact(axes)

# ----------------------------
# For each date being plotted...
# ----------------------------
calCol = ['darkred', 'palegreen']
calColI = 0
count = 0
cObj = ''
tLoc = [] # array for x-axis tick locations to mark the dates on the plot
for d in dateList:
    # Find an oifits directory for this date:
    oiDirs = []
    for dd in dirList:
        if d in dd:
            oiDirs.append(dd)
        if d == '2018Oct25':
            oiDirs = ['2018Oct25_ncoh5ncs1nbs4snr2p0bbiasF'] 
    
    oi,i = 0,0
    while oi == 0:
        try:
            hdrs = mrx.headers.loaddir(oiDirs[i]+'/oifits') # IndexError raised if i exceeds len(oiDirs)
            if hdrs != []:
                # once hdrs are found and read in, break the while loop
                oi += 1
            else:
                # if an oifits directory does not exist in that directory, 
                # check another directory for the same obs date
                i += 1
        except IndexError:
            log.error('Directory "oifits" not found for date '+d)
            log.info('Skipped date '+d)

    # sort the headers by time:
    ids = np.argsort([h['MJD-OBS'] for h in hdrs])
    hdrs = [hdrs[i] for i in ids]
    log.info('Sorted headers by observation date')
    
    # Keep only the calibrator stars?:
    if argopt.only_reference == 'TRUE':
        hdrs = [h for h in hdrs if h['OBJECT'].replace('_',' ') in calL]
        log.info('Cropped SCI targets from header list')
    
    # Check if transmission information has already been saved to the header:
    for b in range(6):
        try:
            bandF = np.append(bf, headers.getval(hdrs,HMQ+'TRANS%i'%b))
        except NameError:
            bandF = headers.getval(hdrs,HMQ+'TRANS%i'%b,default='no')
    
    if 'no' in bandF:
        log.info('Calculate transmission information')
        # Read in the data:
        objList = list(set([h['OBJECT'] for h in hdrs]))
        objCat = dict()
        for obj in objList:
            try:
                cat = Vizier.query_object(obj, catalog='JSDC')[0] # IndexError raised if object not found
                log.info('Find JSDC for '+obj+':')
                log.info(' diam = %.3f mas'%cat['UDDH'][0])
                log.info(' Hmag = %.3f mas'%cat['Hmag'][0])
                objCat[obj] = cat
            except IndexError:
                log.info('Cannot find JSDC for '+obj)
        
        kl = 0 # dummy variable used to ensure that info message is only printed to log once per date
        log.info('Extract camera settings from headers')
        log.info('Calculate transmission on each beam')
        for h in hdrs:
            expT = h['EXPOSURE']
            bWid = h['BANDWID']
            gain = 0.5 * h['GAIN']
            
            try:
                # if info for this object was returned from JSDC:
                Hmag    = float(objCat[h['OBJECT']]['Hmag'][0]) # raises NameError if nothing was returned from JSDC
                fH      = Hzp * 10**(-Hmag/2.5)
                fExpect = fH * expT * bWid * telArea * iTQE
                
                # loop over beams:
                for b in range(6):
                    fMeas = h[HMQ+'BANDFLUX%i MEAN'%b] / gain  # raises KeyError if reduction was done before this keyword was introduced
                    h[HMQ+'TRANS%i'%b] = 100. * (fMeas / fExpect)
            
            except NameError:
                # if info for the object was NOT returned from JSDC:
                for b in range(6):
                    h[HMQ+'TRANS%i'%b] = -1.0
            except KeyError:
                # if info was returned but the reduction is old:
                for b in range(6):
                    h[HMQ+'TRANS%i'%b] = -1.0
                if kl == 0:
                    log.info('QC parameter BANDFLUX missing from header.')
                    log.info('Re-running the reduction is recommended.')
                    kl += 1
    
    # assign colours to data based on SCI or CAL ID and add data to plot:
    countmin = count
    for h in hdrs:
        objname = headers.getval([h],'OBJECT')[0]
        r0      = headers.getval([h],'R0')[0]
        if objname.replace('_', ' ') in calL and objname == cObj:
            # cal is the same as previous so colour must be maintained
            col = calCol[calColI]
            mkr = 'o'
        elif objname.replace('_', ' ') in calL and objname != cObj:
            # cal is different to previous so colour must be changed
            try:
                tcol = calCol[calColI+1]
                calColI += 1
            except:
                calColI += -1
            
            col = calCol[calColI]
            mkr = 'o'
            cObj = objname
        else:
            # target is sci, not cal
            col = 'k'
            mkr = '+'
        # plot the seeing data:
        axes.flatten()[0].plot(count,r0,marker=mkr,color=col,ls='None',ms=5)
        # plot the transmission data:
        for b in range(6):
            transm = headers.getval([h], HMQ+'TRANS%i'%b)
            if transm > 0:
                axes.flatten()[b+1].plot(count, transm, marker=mkr, color=col, ls='None', ms=5)
            try:
                if transm > transmax:
                    transmax = max(transm)
            except NameError:
                transmax = max(transm)
        
        count += 1
        
        del col, mkr, transm, objname
    
    countmax = count
    # add vertical line to plot:
    for b in range(7):
        axes.flatten()[b].plot([count,count],[-0.1,18],ls='-.',color='k')
    count += 1
    
    tLoc.append(int(np.ceil((countmax-countmin)/2))+countmin)
    del countmin, countmax
    
    del hdrs, oiDirs

# -------------------------
# edit the tick parameters and locations:
# -------------------------
axes.flatten()[0].set_title('Mean seeing [10m average]')
axes.flatten()[1].set_title('Transmission [$\%$ of expected $F_\star$]')
axes.flatten()[5].set_xticks(tLoc)
axes.flatten()[5].set_xticklabels(dateList,rotation=70, fontsize=12)

# -------------------------
# save the figure:
# -------------------------
plt.tight_layout()
plt.show()
#files.write (fig,'overview_transmission_'+dateList[0]+'_'+dateList[-1]+'.png')