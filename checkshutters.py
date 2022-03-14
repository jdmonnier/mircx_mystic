import numpy as np;
import os;

import matplotlib.pyplot as plt;
import matplotlib.colors as mcolors;
import mircx_mystic as mrx;
from astropy.stats import sigma_clipped_stats;
from astropy.io import fits as pyfits;
from astropy.modeling import models, fitting;

from skimage.feature import register_translation;

from scipy import fftpack;
from scipy.signal import medfilt;
from scipy.ndimage.interpolation import shift as subpix_shift;
from scipy.ndimage import gaussian_filter;
from scipy.optimize import least_squares;

from . import log, files, headers, setup, oifits, signal, plot;
from .headers import HM, HMQ, HMP, HMW, rep_nan;


def bgkeys (phdrs):

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
        plt.title(key)
        plt.plot(bgtemp[30,280,:])
        #plt.show()
        plt.show()
        #plt.plot(cube[0,:,10,20])
        medbg = np.median(bgtemp,axis=2)
        bgarrays[key] = medbg
        #ig=px.imshow(bgtemp[:,:,0]-medbg)
        #fig.show()
        print('finish plt')
    return bgarrays, keys


def allshutterkeys (phdrs):
    # Group backgrounds for each (gain, conf_na)
    all_phdrs = phdrs # phdrs.loc[phdrs['FILETYPE'] =='BACKGROUND'] # select only Background
    all_hdrs= mrx.headers.p2h(all_phdrs) #JDM replace.
    #bgfiles_gps=bg_phdrs.groupby(by=keys)['ORIGNAME'].apply(list)
    #for bgfiles in bgfiles_gps:
    #    for file in bgfiles:

    keys = ['FILETYPE','CONF_NA','GAIN','NLOOPS','NREADS']
    all_pgps = phdrs.groupby(by=keys)
    all_dict = all_pgps.indices
    keylist=list(all_dict.keys())
    allarrays={}
    for key in keylist: # loop over all the key groups found. 
        print(key)
        print(all_dict[key])
        tuple_keys=['NAXIS4','NAXIS3','NAXIS2','NAXIS1']
        #dimx,dimy=bg_hdrs[bg_dict[key][0]]['NAXIS1'] , bg_hdrs[bg_dict[key][0]]['NAXIS2']
        #DIMX=bg_hdrs[bg_dict[key][0]]['NAXIS2']
        nramps,nframes,dimx,dimy=[all_hdrs[all_dict[key][0]][temp0] for temp0 in tuple_keys] 
        alltemp = np.zeros([dimx,dimy,len(all_dict[key])])
        gaintest=np.zeros(len(all_dict[key]))
        for i,file in enumerate(all_dict[key]): 
            hdr0=[all_hdrs[file]] # pass a list of 1 to next code.
            #JDM make load_raw verson that works with pheaders!
            hdrcheck,cube,__ = files.load_raw (hdr0, coaddRamp='mean',
                                removeBias=False,differentiate=False,
                                saturationThreshold=None,
                                continuityThreshold=None,
                                linear=False,badpix=None,flat=None);
            nframes=hdrcheck['NAXIS3']
            nbin=hdrcheck['NBIN'] #JDM not debugged.
            if nframes < 4:
                breakpoint # will fail if frames per reset <4
            alltemp[:,:,i] = (cube[0,-2,:,:]-cube[0,1,:,:])/(nframes-3.)/nbin
            gaintest[i]=hdrcheck['NAXIS3']
            #plt.plot(cube[0,:,10,20])
            #plt.clf()

            print(file)
        alltemp.shape
        plt.clf()
        plt.plot(alltemp[10,100,:])
        plt.title(key)
        plt.plot(alltemp[30,280,:])
        #plt.show()
        plt.show()
        #plt.plot(cube[0,:,10,20])
        medbg = np.median(alltemp,axis=2)
        allarrays[key] = medbg
        #ig=px.imshow(bgtemp[:,:,0]-medbg)
        #fig.show()
        #print('finish plt')
    return allarrays, keys