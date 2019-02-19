import numpy as np;
import os;

import matplotlib.pyplot as plt;
import matplotlib.colors as mcolors;

from astropy.stats import sigma_clipped_stats;
from astropy.io import fits as pyfits;
from astropy.modeling import models, fitting;

from skimage.feature import register_translation;

from scipy import fftpack;
from scipy.signal import medfilt;
from scipy.ndimage.interpolation import shift as subpix_shift;
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter;
from scipy.optimize import least_squares, curve_fit;
from scipy.ndimage.morphology import binary_closing, binary_opening;
from scipy.ndimage.morphology import binary_dilation, binary_erosion;

from . import log, files, headers, setup, oifits, signal, plot, qc;
from .headers import HM, HMQ, HMP, HMW, rep_nan;

from lmfit import minimize, Minimizer, Parameters;

def bispectrum_minimizer (params,bs,photo,tri_sumv2):
    c0 = params['c0'];
    c1 = params['c1'];
    c2 = params['c2'];

    bs_model = c0 + c1*photo + c2*tri_sumv2;
    resid = bs - bs_model;
    
    return resid;

def compute_bbias_coeff (hdrs, bkgs, fgs, ncoher, output='output_bbias', filetype='BBIAS_COEFF'):
    '''
    Compute the BBIAS_COEFF
    '''
    elog = log.trace ('compute_bbias_coeff');

    # Check inputs
    headers.check_input (hdrs, required=1);
    headers.check_input (bkgs, required=1);
    headers.check_input (fgs, required=1);

    # Build a header for the product
    hdr = hdrs[0].copy();

    # Loop on DATA,FG,BG files
    for ih,(h1,h2,h3) in enumerate(zip(hdrs,bkgs,fgs)):
        # filename
        f1 = h1['ORIGNAME'];
        f2 = h2['ORIGNAME'];
        f3 = h3['ORIGNAME'];
        
        # Load all_dft data and photometry
        log.info ('Load DATA_RTS file %i over %i (%s)'%(ih+1,len(hdrs),f1));
        all_dft_data  = pyfits.getdata (f1, 'ALL_DFT_IMAG').astype(float) * 1.j;
        all_dft_data += pyfits.getdata (f1, 'ALL_DFT_REAL').astype(float);
        data_photo  = pyfits.getdata (f1, 'PHOTOMETRY').astype(float);
        
        log.info ('Load BACKGROUND_RTS file %i over %i (%s)'%(ih+1,len(hdrs),f2));
        all_dft_bg  = pyfits.getdata (f2, 'ALL_DFT_IMAG').astype(float) * 1.j;
        all_dft_bg += pyfits.getdata (f2, 'ALL_DFT_REAL').astype(float);
        bg_photo  = pyfits.getdata (f2, 'PHOTOMETRY').astype(float);

        log.info ('Load FOREGROUND_RTS file %i over %i (%s)'%(ih+1,len(hdrs),f3));
        all_dft_fg  = pyfits.getdata (f3, 'ALL_DFT_IMAG').astype(float) * 1.j;
        all_dft_fg += pyfits.getdata (f3, 'ALL_DFT_REAL').astype(float);
        fg_photo  = pyfits.getdata (f3, 'PHOTOMETRY').astype(float);

        # Form list of closing triangles
        b0=75;
        b1=99;
        tri_list = [];
        for i in np.arange(6,b0-2,3):
            for j in np.arange(4,b1+1,3):
                k=i+j;
                if k>=b0 and k<=b1:
                    tri_list.append([i,j,k]);
        tri_list = np.array(tri_list)
        tri_list = tri_list - 1 #python indices

        # Smooth DATA,BG,FG
        log.info('NCOHERENT %s'%ncoher)
        new_frms = np.arange(int(np.size(all_dft_data,1)/ncoher))*ncoher+1

        all_dft_data = signal.uniform_filter_cpx (all_dft_data,(0,ncoher,0,0),mode='constant');
        data_photo = signal.uniform_filter (data_photo,(0,ncoher,0,0),mode='constant');
        all_dft_bg = signal.uniform_filter_cpx (all_dft_bg,(0,ncoher,0,0),mode='constant');
        bg_photo = signal.uniform_filter (bg_photo,(0,ncoher,0,0),mode='constant');
        all_dft_fg = signal.uniform_filter_cpx (all_dft_fg,(0,ncoher,0,0),mode='constant');
        fg_photo = signal.uniform_filter (fg_photo,(0,ncoher,0,0),mode='constant');

        # Add up all 6 channels for photometry
        data_photo = np.sum(data_photo,axis=-1,keepdims=True);
        fg_photo = np.sum(fg_photo,axis=-1,keepdims=True);
        bg_photo = np.sum(bg_photo,axis=-1,keepdims=True);
        
        # Resample photometry
        data_photo = data_photo[:,new_frms,:,:]#*ncoher
        fg_photo = fg_photo[:,new_frms,:,:]#*ncoher
        bg_photo = bg_photo[:,new_frms,:,:]#*ncoher

        # Combine FG and BG
        fg_photo = np.concatenate([fg_photo,bg_photo])
        all_dft_fg = np.concatenate([all_dft_fg,all_dft_bg])

        # Compute unbiased visibility of DATA
        # based on cross-spectrum with 1-shift
        log.info('Compute unbiased visibility of DATA');
        data_xps = np.real (all_dft_data[:,1:,:,:] * np.conj(all_dft_data[:,:-1,:,:]));
        #data_xps = data_xps[:,new_frms,:,:]#*ncoher*ncoher
        data_xps0 = np.mean(data_xps[:,:,:,b0:b1+1],axis=-1,keepdims=True);
        data_xps = data_xps - data_xps0;

        # Compute unbiased visibility of FG
        #log.info('Compute unbiased visibility of FOREGROUND');
        fg_xps = np.real (all_dft_fg[:,1:,:,:] * np.conj(all_dft_fg[:,:-1,:,:]));
        fg_xps = fg_xps[:,new_frms,:,:]#*ncoher*ncoher
        fg_xps0 = np.mean(fg_xps[:,:,:,b0:b1+1],axis=-1,keepdims=True)
        fg_xps = fg_xps - fg_xps0;

        # Now loop through triangles to compute bispectrum and sum of v2
        log.info('Compute bispectrum and v2 sum of bias closing triangles');
        data_bs=[];
        fg_bs=[];
        data_tri_sumv2=[];
        fg_tri_sumv2=[];
        for tri in tri_list:
            # DATA 
            t_cpx = (all_dft_data)[:,:,:,tri];
            t_cpx = t_cpx[:,:,:,0] * t_cpx[:,:,:,1] * np.conj (t_cpx[:,:,:,2]);
            sumv2 = (data_xps)[:,:,:,tri];
            sumv2 = sumv2[:,:,:,0]+sumv2[:,:,:,1]+sumv2[:,:,:,2];
            data_bs.append(t_cpx);
            data_tri_sumv2.append(sumv2);
            # FOREGROUND 
            t_cpx = (all_dft_fg)[:,:,:,tri];
            t_cpx = t_cpx[:,:,:,0] * t_cpx[:,:,:,1] * np.conj (t_cpx[:,:,:,2]);
            sumv2 = (fg_xps)[:,:,:,tri];
            sumv2 = sumv2[:,:,:,0]+sumv2[:,:,:,1]+sumv2[:,:,:,2];
            fg_bs.append(t_cpx);
            fg_tri_sumv2.append(sumv2);
        data_bs=np.moveaxis(np.array(data_bs),0,-1);
        fg_bs=np.moveaxis(np.array(fg_bs),0,-1);
        data_tri_sumv2=np.moveaxis(np.array(data_tri_sumv2),0,-1);
        fg_tri_sumv2=np.moveaxis(np.array(fg_tri_sumv2),0,-1);

        # Prune measured bispectrum
        data_bs = data_bs[:,new_frms,:,:]#*ncoher*ncoher*ncoher
        fg_bs = fg_bs[:,new_frms,:,:]#*ncoher*ncoher*ncoher

        # Prepare data for minimizer
        fg_photo = fg_photo[:,:,:,:];
        fg_bs = fg_bs[:,:,:,:];
        fg_tri_sumv2 = fg_tri_sumv2[:,:,:,:];
        data_photo = data_photo[:,:,:,:];
        data_bs = data_bs[:,:,:,:];
        data_tri_sumv2 = data_tri_sumv2[:,:,:,:];

        # Average over Frms
        data_bs = np.mean(data_bs,axis=1)
        fg_bs = np.mean(fg_bs,axis=1)
        data_photo = np.mean(data_photo,axis=1)
        fg_photo = np.mean(fg_photo,axis=1)
        data_tri_sumv2 = np.mean(data_tri_sumv2,axis=1)
        fg_tri_sumv2 = np.mean(fg_tri_sumv2,axis=1)

        # Average over ramps
        nramps=15
        new_frms_data = np.arange(int(np.size(data_bs,0)/nramps))*nramps
        new_frms_fg = np.arange(int(np.size(fg_bs,0)/nramps))*nramps

        data_bs = signal.uniform_filter_cpx (data_bs,(nramps,0,0),mode='constant');
        fg_bs = signal.uniform_filter_cpx (fg_bs,(nramps,0,0),mode='constant');
        data_photo = signal.uniform_filter (data_photo,(nramps,0,0),mode='constant');
        fg_photo = signal.uniform_filter (fg_photo,(nramps,0,0),mode='constant');
        data_tri_sumv2 = signal.uniform_filter (data_tri_sumv2,(nramps,0,0),mode='constant');
        fg_tri_sumv2 = signal.uniform_filter (fg_tri_sumv2,(nramps,0,0),mode='constant');

        data_bs = data_bs[new_frms_data,:,:]#*ncoher
        fg_bs = fg_bs[new_frms_fg,:,:]#*ncoher
        data_photo = data_photo[new_frms_data,:,:]#*ncoher
        fg_photo = fg_photo[new_frms_fg,:,:]#*ncoher
        data_tri_sumv2 = data_tri_sumv2[new_frms_data,:,:]#*ncoher
        fg_tri_sumv2 = fg_tri_sumv2[new_frms_fg,:,:]#*ncoher

        ## Avg over triangles
        data_bs = np.mean(data_bs,axis=-1)
        fg_bs = np.mean(fg_bs,axis=-1)
        data_photo = data_photo[:,:,0]
        fg_photo = fg_photo[:,:,0]
        data_tri_sumv2 = np.mean(data_tri_sumv2,axis=-1)
        fg_tri_sumv2 = np.mean(fg_tri_sumv2,axis=-1)
        
        bs = np.concatenate([fg_bs,data_bs]);
        photo = np.concatenate([fg_photo,data_photo]);
        tri_sumv2 = np.concatenate([fg_tri_sumv2,data_tri_sumv2]);
        log.info(bs.shape)
        log.info(photo.shape)
        log.info(tri_sumv2.shape)

        # Fit to model - each spectral channel
        log.info ('Fit coefficients');
        C0 = []
        C1 = []
        C2 = []
        for i in np.arange(np.size(bs,-1)):
            params = Parameters();
            params.add('c0',value=1e6);
            params.add('c1',value=1e3);
            params.add('c2',value=10);
            minner = Minimizer(bispectrum_minimizer,params,fcn_args=(bs.real[:,i],photo[:,i],tri_sumv2[:,i]),nan_policy='omit');
            result = minner.minimize();
            C0.append(result.params['c0'].value);
            C1.append(result.params['c1'].value);
            C2.append(result.params['c2'].value);
        C0 = np.array (C0);
        C1 = np.array (C1);
        C2 = np.array (C2);

        # Figures
        log.info ('Figures');
        fig,ax = plt.subplots ();
        fig.suptitle ('Foreground Data');
        ax.plot(np.ndarray.flatten(fg_photo),np.ndarray.flatten(fg_bs),'.')
        ax.set_xlabel('Photometry')
        ax.set_ylabel('Bispectrum')
        files.write (fig,output+'_fgbispec.png');

        fig,ax = plt.subplots ();
        fig.suptitle ('Foreground Data');
        ax.plot(np.ndarray.flatten(fg_tri_sumv2),np.ndarray.flatten(fg_bs),'.')
        ax.set_ylabel('FG BS')
        ax.set_xlabel('FG SumV2')
        files.write (fig,output+'_fgsumv2.png');

        # File
        log.info ('Create file');

        # First HDU
        hdu0 = pyfits.PrimaryHDU ([]);
        hdu0.header = hdr;
        hdu0.header['FILETYPE'] = filetype;
        hdu0.header[HMQ+'QUALITY'] = (1.0, 'quality of data');

        # Other HDU
        hdu1 = pyfits.ImageHDU (C0);
        hdu1.header['EXTNAME'] = ('C0','Coefficient 1');

        hdu2 = pyfits.ImageHDU (C1);
        hdu2.header['EXTNAME'] = ('C1','Coefficient 2');
        hdu2.header['BUNIT'] = 'adu';

        hdu3 = pyfits.ImageHDU (C2);
        hdu3.header['EXTNAME'] = ('C2','Coefficient 3');
        hdu3.header['BUNIT'] = 'adu';

        # Write file
        hdulist = pyfits.HDUList ([hdu0,hdu1,hdu2,hdu3]);
        files.write (hdulist, output+'.fits');

        plt.close ("all");
        return hdulist;

    
