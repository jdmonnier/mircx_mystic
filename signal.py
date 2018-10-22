import numpy as np;

import scipy;
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter;

from . import log, files, headers, setup, oifits;

def airy (x):
    ''' Airy function, with its zero at x = 1.22'''
    return 2.*scipy.special.jn (1,np.pi*x) / (np.pi*x);
    
def gaussian_filter_cpx (input,sigma,**kwargs):
    ''' Gaussian filter of a complex array '''
    return gaussian_filter (input.real,sigma,**kwargs) + \
           gaussian_filter (input.imag,sigma,**kwargs) * 1.j;

def uniform_filter_cpx (input,sigma,**kwargs):
    ''' Uniform filter of a complex array '''
    return uniform_filter (input.real,sigma,**kwargs) + \
           uniform_filter (input.imag,sigma,**kwargs) * 1.j;
           
def getwidth (curve, threshold=None):
    '''
    Compute the width of curve around its maximum,
    given a threshold. Return the tuple (center,fhwm)
    '''
    
    if threshold is None:
        threshold = 0.5*np.max (curve);

    # Find rising point
    f = np.argmax (curve > threshold) - 1;
    if f == -1:
        log.warning ('Width detected outside the spectrum');
        first = 0;
    else:
        first = f + (threshold - curve[f]) / (curve[f+1] - curve[f]);
    
    # Find lowering point
    l = len(curve) - np.argmax (curve[::-1] > threshold) - 1;
    if l == len(curve)-1:
        log.warning ('Width detected outside the spectrum');
        last = l;
    else:
        last = l + (threshold - curve[l]) / (curve[l+1] - curve[l]);
    
    return 0.5*(last+first), 0.5*(last-first);

def bootstrap_matrix (snr, gd):
    '''
    Compute the best SNR and GD of each baseline when considering 
    also the boostraping capability of the array.
    snr and gd shall be of shape (...,nb)

    Return (snr_b, gd_b) of same size, but including bootstrap.
    '''
    log.info ('Bootstrap baselines with linear matrix');

    # User a power to implement a type of min/max of SNR
    power = 4.0;

    # Reshape
    shape = snr.shape;
    snr = snr.reshape ((-1,shape[-1]));
    gd  = gd.reshape ((-1,shape[-1]));
    ns,nb = gd.shape;

    # Ensure no zero and no nan
    snr[~np.isfinite (snr)] = 0.0;
    snr = np.maximum (snr,1e-1);
    snr = np.minimum (snr,1e3);

    log.info ('Compute OPD_TO_OPD');
    
    # The OPL_TO_OPD matrix
    OPL_TO_OPD = setup.beam_to_base;
    
    # OPD_TO_OPL = (OPL_TO_OPD^T.snr.OPL_TO_OPD)^-1 . OPL_TO_OPD^T.W_OPD
    # o is output OPL
    JtW = np.einsum ('tb,sb->stb',OPL_TO_OPD.T,snr**power);
    JtWJ = np.einsum ('stb,bo->sto',JtW,OPL_TO_OPD);

    JtWJ_inv = np.array([ np.linalg.pinv (JtWJ[s]) for s in range(ns)]);# 'sot'
    OPD_TO_OPL = np.einsum ('sot,stb->sob', JtWJ_inv, JtW);

    # OPD_TO_OPD = OPL_TO_OPD.OPD_TO_OPL  (m is output OPD)
    OPD_TO_OPD = np.einsum ('mo,sob->smb', OPL_TO_OPD, OPD_TO_OPL);
    
    log.info ('Compute gd_b and snr_b');
    
    # GDm = OPD_TO_OPD . GD
    gd_b = np.einsum ('smb,sb->sm',OPD_TO_OPD,gd);

    # Cm = OPD_TO_OPD . C_OPD . OPD_TO_OPD^T
    OPD_TO_OPD_W = np.einsum ('smb,sb->smb',OPD_TO_OPD,snr**-power);
    cov_b = np.einsum ('smb,snb->smn',OPD_TO_OPD_W, OPD_TO_OPD);

    # Reform SNR from covariance
    snr_b = np.diagonal (cov_b, axis1=1, axis2=2)**-(1./power);
    snr_b[snr_b < 1e-2] = 0.0;
    
    # Reshape
    snr = snr.reshape (shape);
    gd  = gd.reshape (shape);
    snr_b = snr_b.reshape (shape);
    gd_b  = gd_b.reshape (shape);

    return (snr_b,gd_b);

def bootstrap_triangles (snr,gd):
    '''
    Compute the best SNR and GD of each baseline when considering 
    also the boostraping capability of the array.
    snr and gd shall be of shape (...,nb)

    Return (snr_b, gd_b) of same size, but including bootstrap.
    '''

    log.info ('Bootstrap baselines with triangles');

    # Reshape
    shape = snr.shape;
    snr = snr.reshape ((-1,shape[-1]));
    gd  = gd.reshape ((-1,shape[-1]));
    ns,nb = gd.shape;

    # Ensure no zero and no nan
    snr[~np.isfinite (snr)] = 0.0;
    snr = np.maximum (snr,1e-1);
    snr = np.minimum (snr,1e3);
    
    # Create output
    gd_b  = gd.copy ();
    snr_b = snr.copy ();

    # Sign of baseline in triangles
    sign = np.array ([1.0,1.0,-1.0]);

    # Loop several time over triplet to also
    # get the baseline tracked by quadruplets.
    for i in range (7):
        for tri in setup.triplet_base ():
            for s in range (ns):
                i0,i1,i2 = np.argsort (snr_b[s,tri]);
                # Set SNR as the worst of the two best
                snr_b[s,tri[i0]] = snr_b[s,tri[i1]];
                # Set the GD as the sum of the two best
                mgd = gd_b[s,tri[i1]] * sign[i1] + gd_b[s,tri[i2]] * sign[i2];
                gd_b[s,tri[i0]] = - mgd * sign[i0];
                
    # Reshape
    snr = snr.reshape (shape);
    gd  = gd.reshape (shape);
    snr_b = snr_b.reshape (shape);
    gd_b  = gd_b.reshape (shape);
    
    return (snr_b,gd_b);

def psd_projection (scale, freq, freq0, delta0, data):
    '''
    Project the PSD into a scaled theoretical model,
    Return the merit function 1. - D.M / sqrt(D.D*M.M)
    '''

    # Scale the input frequencies
    freq_s = freq * scale;
    
    # Compute the model of the PSD
    model = np.sum (np.exp (- (freq_s[:,None] - freq0[None,:])**2 / delta0**2), axis=-1);

    if data is None:
        return model;

    # Return the merit function from the normalised projection
    weight = np.sqrt (np.sum (model * model) * np.sum (data * data));
    return 1. - np.sum (model*data) / weight;
