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

def compute_bbias_coeff (hdrs, bkgs, fgs, ncoher, output='output_bbias', filetype='BBIAS_COEFF'):
    '''
    Compute the BBIAS_COEFF
    '''
    elog = log.trace ('compute_bbias_coeff');

    # FIXME: hardcode ncoherent for bbias computation (low is less noisy)
    ncoher = 3

    # Check inputs
    headers.check_input (hdrs, required=1);
    headers.check_input (bkgs, required=1);
    headers.check_input (fgs, required=1);

    # Build a header for the product
    hdr = hdrs[0].copy();

    # Form list of closing triangles
    b0=75;
    b1=99;
    tri_list = [];
    for i in np.arange(6,b0-6,3):
        for j in np.arange(i,b1-3,3):
            k=i+j;
            if k>=b0 and k<=b1:
                tri_list.append([i,j,k]);
    tri_list = np.array(tri_list);

    # Loop on DATA,FG,BG files
    all = hdrs + fgs + bkgs;
    bispectrum = None;
    for ih,h1 in enumerate(all):
        # filename
        f1 = h1['ORIGNAME'];
        
        # Load all_dft data
        log.info ('Load RTS file %i over %i (%s)'%(ih+1,len(all),f1));
        all_dft  = pyfits.getdata (f1, 'ALL_DFT_IMAG').astype(float) * 1.j;
        all_dft += pyfits.getdata (f1, 'ALL_DFT_REAL').astype(float);

        # Load all_dft photometry
        photo  = pyfits.getdata (f1, 'PHOTOMETRY').astype(float);

        # Smooth DATA,PHOTO
        log.info('NCOHERENT %s'%ncoher);
        new_frms = np.arange(int(np.size(all_dft,1)/ncoher))*ncoher+1;

        all_dft = signal.uniform_filter_cpx (all_dft,(0,ncoher,0,0),mode='constant');
        photo = signal.uniform_filter (photo,(0,ncoher,0,0),mode='constant');

        # Add up all 6 channels for photometry
        photo = np.sum(photo,axis=-1,keepdims=True);
        
        # Resample photometry
        photo = photo[:,new_frms,:,:]*ncoher;

        # Compute unbiased visibility of DATA
        # based on cross-spectrum with 1-shift
        log.info('Compute unbiased visibility of DATA');
        data_xps = np.real (all_dft[:,1:,:,:] * np.conj(all_dft[:,:-1,:,:]));
        data_xps = data_xps[:,new_frms,:,:]*ncoher*ncoher;
        data_xps0 = np.mean(data_xps[:,:,:,b0:b1+1],axis=-1,keepdims=True);
        data_xps = data_xps - data_xps0;

        # Now loop through triangles to compute bispectrum and sum of v2
        # FIXME append
        log.info('FIXME: Compute bispectrum and v2 sum of bias closing triangles');
        bs=[];
        tri_sumv2=[];
        for tri in tri_list:
            # DATA 
            t_cpx = (all_dft)[:,:,:,tri];
            t_cpx = t_cpx[:,:,:,0] * t_cpx[:,:,:,1] * np.conj (t_cpx[:,:,:,2]);
            sumv2 = (data_xps)[:,:,:,tri];
            sumv2 = sumv2[:,:,:,0]+sumv2[:,:,:,1]+sumv2[:,:,:,2];
            bs.append(t_cpx);
            tri_sumv2.append(sumv2);
        bs=np.moveaxis(np.array(bs),0,-1);
        tri_sumv2=np.moveaxis(np.array(tri_sumv2),0,-1);

        # Resample measured bispectrum
        bs = bs[:,new_frms,:,:]*ncoher*ncoher*ncoher;

        # Average over Frms
        bs = np.mean(bs,axis=1);
        photo = np.mean(photo,axis=1);
        tri_sumv2 = np.mean(tri_sumv2,axis=1);

        # Average over ramps
        nramps=15;
        if np.size(bs,0)<nramps:
            nramps = np.size(bs,0)
        new_frms_data = np.arange(int(np.size(bs,0)/nramps))*nramps+1;

        bs = signal.uniform_filter_cpx (bs,(nramps,0,0),mode='constant');
        photo = signal.uniform_filter (photo,(nramps,0,0),mode='constant');
        tri_sumv2 = signal.uniform_filter (tri_sumv2,(nramps,0,0),mode='constant');

        bs = bs[new_frms_data,:,:];
        photo = photo[new_frms_data,:,:];
        tri_sumv2 = tri_sumv2[new_frms_data,:,:];

        ## Avg over triangles
        bs = np.mean(bs,axis=-1);
        photo = photo[:,:,0];
        tri_sumv2 = np.mean(tri_sumv2,axis=-1);

        ## Take median over channels
        #nx,ny = bs.shape
        #bs = np.median(bs,axis=-1,keepdims=True)
        #photo = np.median(photo,-1,keepdims=True)
        #tri_sumv2 = np.median(tri_sumv2,axis=-1,keepdims=True)
        #bs = np.repeat(bs,ny,axis=-1)
        #photo = np.repeat(photo,ny,axis=-1)
        #tri_sumv2 = np.repeat(tri_sumv2,ny,axis=-1)

        if bispectrum is None:
            bispectrum = np.empty((0,bs.shape[1]),int);
            photometry = np.empty((0,photo.shape[1]),int);
            sum_vis2 = np.empty((0,tri_sumv2.shape[1]),int);

        bispectrum=np.append(bispectrum,bs,axis=0);
        photometry=np.append(photometry,photo,axis=0);
        sum_vis2=np.append(sum_vis2,tri_sumv2,axis=0);

        log.info ('Cleanup memory');
        del bs, photo, tri_sumv2, all_dft, data_xps, data_xps0, t_cpx, sumv2;

    ## measure coefficients
    log.info('Measure bbias coefficients');
    C0 = [];
    C1 = [];
    C2 = [];
    for i in np.arange(np.size(bispectrum,-1)):
        p = photometry[:,i];
        b = bispectrum[:,i].real;
        s = sum_vis2[:,i];
        
        if np.isfinite(p).all()==False:
            C0.append(np.nan);
            C1.append(np.nan);
            C2.append(np.nan);
        else:
            # median filter to deal with outliers
            b = medfilt(b,5);
            s = medfilt(s,5);
            p = medfilt(p,5);

            # Measure coefficients
            unit = p*0. + 1.;
            A = np.array([unit,p,s]);
            result = np.linalg.lstsq(A.T,b);
            C0.append(result[0][0]);
            C1.append(result[0][1]);
            C2.append(result[0][2]);

            # Residuals 
            bs_model = result[0][0] + result[0][1]*p+ result[0][2]*s;
            resid = (b - bs_model) / ((b+bs_model)/2) * 100;

            # Figures
            fig,ax = plt.subplots ();
            fig.suptitle ('Bispectrum - Photometry');
            ax.plot(p,b,'.');
            ax.set_xlabel('Photometry');
            ax.set_ylabel('Bispectrum');
            files.write (fig,output+'_bispec_%s.png'%i);

            ## check for crazy vis2 measurement
            log.info ('Figures');
            fig,ax = plt.subplots ();
            fig.suptitle ('Bispectrum - SumV2');
            ax.plot(s,b,'.');
            ax.set_xlabel('SumV2');
            ax.set_ylabel('Bispectrum');
            files.write (fig,output+'_sumv2_%s.png'%i);

            ## photometry resids 
            log.info ('Figures');
            fig,ax = plt.subplots ();
            fig.suptitle ('Photometry Residuals');
            ax.plot(p,resid,'.');
            ax.set_xlabel('Photometry');
            ax.set_ylabel('Residual (%)');
            files.write (fig,output+'_photoresid_%s.png'%i);

            ## sumvis2 resids
            fig,ax = plt.subplots ();
            fig.suptitle ('SumV2 Residuals');
            ax.plot(s,resid,'.');
            ax.set_ylabel('Residual (%)');
            ax.set_xlabel('SumV2');
            files.write (fig,output+'_sumv2resid_%s.png'%i);

    C0 = np.array(C0);
    C1 = np.array(C1);
    C2 = np.array(C2);

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