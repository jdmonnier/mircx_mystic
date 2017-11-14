import numpy as np;
from scipy.ndimage import gaussian_filter;

from . import log, files, headers, setup, oifits;

def gaussian_filter_cpx (input,sigma,**kwargs):
    ''' Gaussian filter of a complex array '''
    return gaussian_filter (input.real,sigma,**kwargs) + \
           gaussian_filter (input.imag,sigma,**kwargs) * 1.j;
               
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
    
    return 0.5*(last+first), 0.5*(last-first)

def bootstrap (snr, gd):
    '''
    snr and gd shall be (...,nb)
    '''
    log.info ('Bootstrap baselines');

    # Reshape
    shape = snr.shape;
    snr = snr.reshape ((-1,shape[-1]));
    gd  = gd.reshape ((-1,shape[-1]));
    ns,nb = gd.shape;

    # Ensure no zero and non-nan
    snr[~np.isfinite (snr)] = 0.0;
    snr += 1e-5;

    log.info ('Compute OPD_TO_OPD');
    
    # The OPL_TO_OPD matrix
    OPL_TO_OPD = setup.beam_to_base;
    
    # OPD_TO_OPL = (OPL_TO_OPD^T.snr.OPL_TO_OPD)^-1 . OPL_TO_OPD^T.W_OPD
    # o is output OPL
    JtW = np.einsum ('tb,sb->stb',OPL_TO_OPD.T,snr);
    JtWJ = np.einsum ('stb,bo->sto',JtW,OPL_TO_OPD);

    JtWJ_inv = np.array([ np.linalg.pinv (JtWJ[s]) for s in range(ns)]); # 'sot'
    OPD_TO_OPL = np.einsum ('sot,stb->sob', JtWJ_inv, JtW);

    # OPD_TO_OPD = OPL_TO_OPD.OPD_TO_OPL  (m is output OPD)
    OPD_TO_OPD = np.einsum ('mo,sob->smb', OPL_TO_OPD, OPD_TO_OPL);
    
    log.info ('Compute gd_b and snr_b');
    
    # GDm = OPD_TO_OPD . GD
    gd_b = np.einsum ('smb,sb->sm',OPD_TO_OPD,gd);

    # Cm = OPD_TO_OPD . C_OPD . OPD_TO_OPD^T
    OPD_TO_OPD_W = np.einsum ('smb,sb->smb',OPD_TO_OPD,1./snr);
    cov_b = np.einsum ('smb,snb->smn',OPD_TO_OPD_W, OPD_TO_OPD);

    # Reform SNR from covariance
    snr_b = 1. / np.diagonal (cov_b, axis1=1, axis2=2);
    snr_b[snr_b < 1e-3] = 0.0;
    
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
