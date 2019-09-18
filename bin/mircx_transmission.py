#! /usr/bin/env python                                                          
# -*- coding: iso-8859-15 -*-                                                   

import mircx_pipeline as mrx
import argparse, glob, os, sys
import datetime as dattime
from datetime import datetime
import numpy as np
from astropy.io import fits as pyfits
import matplotlib.pyplot as plt
import os
from dateutil.parser import parse

from mircx_pipeline import log, headers, plot, files
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

parser.add_argument("--dir", dest="dir",default='/data/MIRCX/reduced',type=str,
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
parser.add_argument("--oifits-dir",dest="oifits_dir",default='.',type=str,
                    help="directory of products [%(default)s]")


# Parse arguments:
argopt = parser.parse_args()

# Verbose:
elog = log.trace('mircx_transmission')
o1 = ' --num-nights='+str(float(argopt.num_of_nights))+' --date-from='+argopt.night_from
o2 = ' --date-to='+argopt.night_to+' --targ-list='+argopt.targ_list
o3 = ' --only-reference='+str(argopt.only_reference)+' --oifits-dir='+argopt.oifits_dir
log.info('Run mircx_transmission.py --dir='+argopt.dir+o1+o2+o3)

# Check how many nights are to be plotted:
now = datetime.now()
if argopt.num_of_nights != 0:
    nNight = argopt.num_of_nights
else:
    if argopt.night_from == '':
        nNight = 14 # default to plotting the 14 most recent nights of data
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

def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    param string: str, string to check for date
    param fuzzy: bool, ignore unknown tokens in string if True
    """
    try: 
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False

# Check the directory structure read in directory names from trunk
if is_date(sDir.split('/')[-2]):
    # we're in a /data/reduced/YYYYMmm/YYYYMmmDD format directory tree
    dirList = glob.glob('/'.join(sDir.split('/')[:-2])+'/*/*')
else:
    # we're in a /data/reduced/YYYYMmmDD format directory tree
    dirList = glob.glob(sDir+'*')

dL = list(set([d.split('_')[0].split('/')[-1] for d in dirList])) # remove duplicate dates
for d in dL:
    try:
        dL1.append(datetime.strptime(d,'%Y%b%d')) # for sorting, translate these into datetime format
    except NameError:
        # first instance:
        dL1 = []
    except ValueError:
        # ensure other things in the directory are skipped over but keep a note of what they are
        log.info('Skipped file in directory: '+d)

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
    # Check that lNight is in the dateList:
    while lNight not in dateList and lNight != now.strftime('%Y%b%d'):
        # increase the day by one until the next obs date or current date is reached:
        today = datetime.strptime(lNight, '%Y%b%d')
        nextDay = today + dattime.timedelta(days=1)
        nD = nextDay.strftime('%Y%b%d')
        lNight = nD
    
    if lNight not in dateList:
        dL3 = dateList[dateList.index(fNight)]
    else:
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
        try:
            if line.split(',')[5] == 'CAL':
                calL.append(line.split(',')[0])
        except IndexError:
            log.info('Final line in localDB file is blank: please fix')

# Load astroquery
try:
    from astroquery.vizier import Vizier;
    log.info ('Load astroquery.vizier');
    from astroquery.simbad import Simbad;
    log.info ('Load astroquery.simbad');
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
fig,axes = plt.subplots(7,1,sharex=True,figsize=(9,16))
plot.compact(axes)

# ----------------------------
# For each date being plotted...
# ----------------------------
calCol = ['darkred', 'palegreen']
calColI = 0
count = 0
cObj = ''
tLoc = [] # array for x-axis tick locations to mark the dates on the plot
oiDir = argopt.oifits_dir
dNames = []
for d in dateList:
    # Find an oifits directory for this date:
    oiDirs = []
    for dd in dirList:
        if d in dd and 'ncoh' not in dd and '.png' not in dd and 'bracket' not in dd:
            oiDirs.append(dd)
        if d == '2018Oct25':
            oiDirs = ['2018Oct25_nbs0ncs1bbiasTmitp30']
    log.info('Found the following data directories for '+d)
    log.info(oiDirs)
    
    oi,i = 0,0
    if oiDirs == []:
        oi += 1 # ensures that the user doesn't get stuck in the while loop
    
    while oi == 0 and i < len(oiDirs):
        if os.path.isdir(oiDirs[i]+'/'+oiDir):
            hdrs = mrx.headers.loaddir(oiDirs[i]+'/'+oiDir)
            if hdrs != []:
                # once hdrs are found and read in, break the while loop
                oi += 1
            else:
                # if an oifits directory does not exist in that directory, 
                # check another directory for the same obs date
                i += 1
        else:
            i += 1
    
    try:
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
            objList[:] = [x for x in objList if x not in ['NOSTAR', '', 'STS']]
            # ^--- removes NOSTAR and blank object name instances from object list
            objCat = dict()
            exclude = ['NOSTAR', '', 'STS']
            for obj in objList:
                try:
                    cat = Vizier.query_object(obj, catalog='JSDC')[0] 
                    # ^-- IndexError raised if object not found
                    log.info('Find JSDC for '+obj+':')
                    ind = list(cat['Name']).index(obj.replace('_', ' '))
                    # ^-- ValueError raised if object name in JSDC is not what we use
                    log.info(' diam = %.3f mas'%cat['UDDH'][ind])
                    log.info(' Hmag = %.3f mas'%cat['Hmag'][ind])
                    objCat[obj] = cat[ind]
                    del ind
                except IndexError:
                    log.info('Cannot find JSDC for '+obj)
                    exclude.append(obj)
                except ValueError:
                    ind = -999
                    # (sometimes we get here when JSDC finds neighbouring stars but not our target)
                    # (other times we get here if the object name in JSDC is an alias)
                    alt_ids = Simbad.query_objectids(obj)
                    for a_id in list(cat['Name']):
                        if a_id in list(alt_ids['ID']):
                            ind = list(cat['Name']).index(a_id)
                        elif a_id in list([a.replace(' ', '') for a in alt_ids['ID']]):
                            ind = list(cat['Name']).index(a_id)
                    if ind != -999:
                        log.info(' diam = %.3f mas'%cat['UDDH'][ind])
                        log.info(' Hmag = %.3f mas'%cat['Hmag'][ind])
                        objCat[obj] = cat[ind]
                    else:
                        log.info('Cannot find JSDC for '+obj)
                        exclude.append(obj)
                    del ind
        
            kl = 0 # dummy variable used to ensure that info message is only printed to log once per date
            log.info('Extract camera settings from headers')
            log.info('Calculate transmission on each beam')
            for h in hdrs:
                if h['OBJECT'] not in exclude:
                    expT = h['EXPOSURE']
                    bWid = abs(h['BANDWID'])
                    gain = 0.5 * h['GAIN']
                    try:
                        Hmag    = float(objCat[h['OBJECT']]['Hmag']) # raises NameError if nothing was returned from JSDC
                        fH      = Hzp * 10**(-Hmag/2.5)
                        fExpect = fH * expT * bWid * telArea * iTQE
                        for b in range(6):
                            fMeas = h[HMQ+'BANDFLUX%i MEAN'%b] / gain  # raises KeyError if reduction was done before this keyword was introduced
                            if fMeas < 0.:
                                h[HMQ+'TRANS%i'%b] = -1.0
                            else:
                                h[HMQ+'TRANS%i'%b] = 100. * (fMeas / fExpect)
                    
                    except NameError:
                        # if info for the object was NOT returned from JSDC:
                        for b in range(6):
                            h[HMQ+'TRANS%i'%b] = -1.0
                    except KeyError:
                        # if info was returned but the reduction is old or object name not in JSDC:
                        for b in range(6):
                            h[HMQ+'TRANS%i'%b] = -1.0
                        if kl == 0:
                            log.info('QC parameter BANDFLUX missing from header.')
                            log.info('Re-running the reduction is recommended.')
                            kl += 1
                else:
                    for b in range(6):
                        h[HMQ+'TRANS%i'%b] = -1.0
        
        # assign colours to data based on SCI or CAL ID and add data to plot:
        countmin = count
        for h in hdrs:
            objname = headers.getval([h],'OBJECT')[0]
            if objname not in exclude:
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
            elif objname != 'NOSTAR' and objname != '' and objname != 'STS':
                # plot the seeing data:
                axes.flatten()[0].plot(count,headers.getval([h],'R0')[0],marker='+',color='k',ls='None',ms=5)
                # don't bother plotting the transmission data cos the values are just '-1'
                count += 1
        
        countmax = count
        # add vertical line to plot:
        for b in range(7):
            axes.flatten()[b].plot([count,count],[-0.1,18],ls='-.',color='k')
        count += 1
    
    
        tLoc.append(int(np.ceil((countmax-countmin)/2))+countmin)
        del countmin, countmax
    
        del hdrs, oiDirs
        dNames.append(d)
    except NameError:
        log.error('No calibrated data found for '+d+'...skipped date')

# -------------------------
# edit the tick parameters and locations:
# -------------------------
for b in range(1, 7):
    axes.flatten()[b].set_ylim([-0.1, transmax])

axes.flatten()[0].set_title('Mean seeing [10m average]')
axes.flatten()[1].set_title('Transmission [$\%$ of expected $F_\star$]')
axes.flatten()[5].set_xticks(tLoc)
axes.flatten()[5].set_xticklabels(dNames,rotation=70, fontsize=12)

# -------------------------
# save the figure:
# -------------------------
plt.tight_layout()
#plt.show()
if dateList[0] != dateList[-1]:
    files.write(fig,sDir+'overview_transmission_'+dNames[0]+'_'+dNames[-1]+'.png')
else:
    files.write(fig,sDir+'transmission_'+dNames[0]+'.png')
