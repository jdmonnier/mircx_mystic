#! /usr/bin/env python                                                          
# -*- coding: iso-8859-15 -*- 

from astropy.io import fits as pyfits
import numpy as np
import glob, sys
import matplotlib.pyplot as plt
from matplotlib import cm
from mircx_pipeline import lookup, summarise, mailfile, headers, log, files

"""
This produces the following plots:
 - uv coverage in m
 - uv coverage in M$\lambda$
 - vis vs spatial freq (coloured by wavelength)
 - vis2 vs spaial freq (coloured by wavelength)
 - cp vs max spatial freq (coloured by wavelength)

To run, use python quickview_mircx.py /full/path/to/calibrated/data

The script expects all files in the folder to be for the same object.

Plots will be saved to the parsed folder.
"""

def get_spfreqs(hdu,name):
    '''
    Return the spatial frequency B/lbd in M$\lambda
    '''
    lbd = hdu['OI_WAVELENGTH'].data['EFF_WAVE']*1e6
    
    if name == 'OI_VIS2':
        u = hdu['OI_VIS2'].data['UCOORD']
        # the sign doesn't matter here because things are squared
        v = hdu['OI_VIS2'].data['VCOORD']
        return [np.sqrt(u**2+v**2)[:,None]/lbd[None,:],-u[:,None]/lbd[None,:],v[:,None]/lbd[None,:]]
    
    if name == 'OI_T3':
        u1 = hdu['OI_T3'].data['U1COORD']
        v1 = hdu['OI_T3'].data['V1COORD']
        u2 = hdu['OI_T3'].data['U2COORD']
        v2 = hdu['OI_T3'].data['V2COORD']
        u = np.array([u1,u2,u1+u2])
        v = np.array([v1,v2,v1+v2])
        return np.sqrt (u**2 + v**2)[:,:,None] / lbd[None,None,:]

def reshape_uv(hdu):
    '''
    Return the u and v coordinates in the same shape as vis2
    '''
    
    dumY = np.ones(np.shape(hdu['OI_WAVELENGTH'].data['EFF_WAVE']))
    u = 0.0-hdu['OI_VIS2'].data['UCOORD']
    v = hdu['OI_VIS2'].data['VCOORD']
    return [u[:,None]/dumY[None,:],v[:,None]/dumY[None,:]]


files = sorted(glob.glob(sys.argv[1]+'*.fits'))
#files = sorted(glob.glob('*.fits'))

# Load first file
hdu    = pyfits.open(files[0])
object = hdu[0].header['OBJECT']
wave   = hdu['OI_WAVELENGTH'].data['EFF_WAVE']*1e6
flag   = hdu['OI_VIS2'].data['FLAG']
vis2   = hdu['OI_VIS2'].data['VIS2DATA']
evis2  = hdu['OI_VIS2'].data['VIS2ERR']
ucoord, vcoord = reshape_uv(hdu)

flag2 = hdu['OI_T3'].data['FLAG']
cp    = hdu['OI_T3'].data['T3PHI']
ecp   = hdu['OI_T3'].data['T3PHIERR']

spf, usf, vsf = get_spfreqs(hdu,'OI_VIS2')
max_sf = np.max(get_spfreqs(hdu,'OI_T3'),axis=0)

# Load other files
for file in files[1:]:
    hdu    = pyfits.open(file);
    flag   = np.append(flag, hdu['OI_VIS2'].data['FLAG'], axis=0)
    vis2   = np.append(vis2, hdu['OI_VIS2'].data['VIS2DATA'], axis=0)
    evis2  = np.append(evis2, hdu['OI_VIS2'].data['VIS2ERR'], axis=0)
    ucoord = np.append(ucoord, reshape_uv(hdu)[0], axis=0)
    vcoord = np.append(vcoord, reshape_uv(hdu)[1], axis=0)
    spf    = np.append(spf, get_spfreqs(hdu,'OI_VIS2')[0], axis=0)
    usf    = np.append(usf, get_spfreqs(hdu,'OI_VIS2')[1], axis=0)
    vsf    = np.append(vsf, get_spfreqs(hdu,'OI_VIS2')[2], axis=0)
    
    flag2  = np.append(flag2, hdu['OI_T3'].data['FLAG'], axis=0)
    cp     = np.append(cp, hdu['OI_T3'].data['T3PHI'], axis=0)
    ecp    = np.append(ecp, hdu['OI_T3'].data['T3PHIERR'], axis=0)
    max_sf = np.append(max_sf, np.max(get_spfreqs(hdu,'OI_T3'),axis=0), axis=0)

# Interpolate flagged values
vis2[flag] = np.nan
evis2[flag] = np.nan
vsf[flag] = np.nan
usf[flag] = np.nan
ucoord[flag] = np.nan
vcoord[flag] = np.nan

cp[flag2] = np.nan
ecp[flag2] = np.nan
max_sf[flag2] = np.nan

copcol = cm.get_cmap('winter', len(wave)) # was 'copper'

#
# Plot squared vis
#
figv2 = plt.figure(1, figsize=(6,4))
axv2  = plt.subplot2grid((1, 1), (0, 0))
for w in range(0, len(wave)):
   axv2.errorbar(spf[:,w],vis2[:,w],yerr=evis2[:,w],fmt='o',ms=1,color=copcol(w))

axv2.set_xlim(0,230)
axv2.set_ylim(0.0,1.0)
axv2.set_xlabel('Baseline (M$\lambda$)')
axv2.set_ylabel('Vis2')
figv2.savefig(sys.argv[1]+'/'+object+'_vis2_sf.pdf')

#
# Plot vis
#
figV = plt.figure(2, figsize=(6,4))
axV  = plt.subplot2grid((1, 1), (0, 0))
for w in range(0, len(wave)):
   evis = 0.5*evis2[:,w]*np.power(vis2[:,w], -0.5)
   axV.errorbar(spf[:,w],np.sqrt(vis2[:,w]),yerr=evis,fmt='o',ms=1,color=copcol(w))

axV.set_xlim(0,230)
axV.set_ylim(0.0,1.0)
axV.set_xlabel('Baseline (M$\lambda$)')
axV.set_ylabel('Vis')
figV.savefig(sys.argv[1]+'/'+object+'_vis_sf.pdf')

#
# Plot CP vs Bmax
#
figCP = plt.figure(3, figsize=(6,4))
axCP  = plt.subplot2grid((1, 1), (0, 0))
#axCP.hlines(0, 0,230, linestyles='dashed', color='grey')
#axCP.hlines(-200, 0,230, linestyles='dashed', color='w')
#axCP.hlines(200, 0,230, linestyles='dashed', color='w')
for w in range(0, len(wave)):
    axCP.errorbar(max_sf[:,w],cp[:,w],yerr=ecp[:,w],fmt='o',ms=1,color=copcol(w))

axCP.set_xlim=(0.,230.)
axCP.set_xlabel('Max baseline (M$\lambda$)')
axCP.set_ylabel('$\phi_{CP}$')
plt.savefig(sys.argv[1]+'/'+object+'_t3phi_maxb.pdf')


#
# Plot UV plane
#
# Recall
#    +u = East
#    +v = North
figUVm = plt.figure(4, figsize=(4,4))
axUVm  = plt.subplot2grid((1, 1), (0, 0))
axUVm.plot(ucoord,vcoord,'o',color='b')
axUVm.plot(-ucoord,-vcoord,'o',color='b')
axUVm.set_xlim(-330,330)
axUVm.set_ylim(-330,330)
axUVm.set_xlabel('u (m)') # $\longleftarrow$ East
axUVm.set_ylabel('v (m)') # North $\longrightarrow$
axUVm.set_aspect(1)
axUVm.grid(True)
figUVm.tight_layout()
figUVm.savefig(sys.argv[1]+'/'+object+'_uv_m.pdf')

figUVl = plt.figure(5, figsize=(4,4))
axUVl  = plt.subplot2grid((1, 1), (0, 0))
axUVl.plot(usf,vsf,'o',color='b')
axUVl.plot(-usf,-vsf,'o',color='b')
axUVl.set_xlim(-230,230)
axUVl.set_ylim(-230,230)
axUVl.set_xlabel('u (M$\lambda$)') # $\longleftarrow$ East
axUVl.set_ylabel('v (M$\lambda$)') # North $\longrightarrow$
axUVl.set_aspect(1)
axUVl.grid(True)
figUVl.tight_layout()
figUVl.savefig(sys.argv[1]+'/'+object+'_uv_Mlambda.pdf')

# Write summary files
localDB = os.environ['MIRCX_PIPELINE']+'/mircx_pipeline/mircx_targets.list'
calInfo, scical = lookup.queryLocal(objlist, localDB)
calF=True
redF=True

log.info('Read headers from raw data directory')
rawhdrs = headers.loaddir(argopt.oifits_dir) 
log.info('Create report summary files')
outfiles = [argopt.oifits_dir+'/summary.tex']

ncoh = headers.getval (rawhdrs, 'NCOHER')[0]
log.info(ncoh)

for outFile in outfiles:
        with open(outFile, 'w') as outtex:
            outtex.write('\\documentclass[a4paper]{article}\n\n')
            outtex.write('\\usepackage{fullpage}\n\\usepackage{amsmath}\n')
            outtex.write('\\usepackage{hyperref}\n\\usepackage{graphicx}\n')
            outtex.write('\\usepackage{longtable}\n')
            #outtex.write('\\usepackage[left=1.5cm,right=1.5cm,top=2cm,bottom=2cm]\n')
            outtex.write('\\begin{document}\n')

summarise.texSumTables(argopt.oifits_dir,objlist,calInfo,scical,redF,rawhdrs,outfiles)
log.info('Cleanup memory')
del rawhdrs
#summarise.texReportPlts(argopt.oifits_dir,outfiles,dates[d])
#summarise.texSumUV(argopt.oifits_dir,calF,outfiles)
summarise.texSumPlots(argopt.oifits_dir,redF,calF,outfiles)
subprocess.call('pdflatex '+outfiles[0], shell=True)
#subprocess.call('pdflatex '+outfiles[0] , shell=True)
