import numpy as np;
from astropy.io import fits as pyfits;

from . import log, headers, setup;
from .headers import HM, HMQ, HMP, HMW, rep_nan;

def flux (hdr, y0, photo):
    '''
    Add QC to hdr about flux
    '''
    
    log.info ('Compute QC for xchan flux');
    
    for t in range(6):
        val = np.mean (photo[:,:,y0,t], axis=(0,1));
        hdr[HMQ+'FLUX%i MEAN'%t] = (val,'flux at lbd0');

def snr (hdr, y0, base_snr0, base_snr):
    '''
    Add QC to hdr about snr.
    '''
    
    log.info ('Compute QC for SNR');
    
    for b,name in enumerate (setup.base_name ()):
        val = rep_nan (np.mean (base_snr0[:,:,:,b]));
        hdr[HMQ+'SNR'+name+' MEAN'] = (val,'Broad-band SNR');
        val = rep_nan (np.mean (base_snr[:,:,:,b]));
        hdr[HMQ+'SNRB'+name+' MEAN'] = (val,'Broad-band bootstrapped SNR');
        

def power (hdr, y0, base_power, bias_power, norm_power):
    '''
    Add QC to hdr about snr.
    '''

    # QC for power
    log.info ('Compute QC for power');
    
    for b,name in enumerate (setup.base_name ()):
        val = rep_nan (np.mean (norm_power[:,:,y0,b], axis=(0,1)));
        hdr[HMQ+'NORM'+name+' MEAN'] = (val,'Norm Power at lbd0');
        val = rep_nan (np.mean (base_power[:,:,y0,b], axis=(0,1)));
        hdr[HMQ+'POWER'+name+' MEAN'] = (val,'Fringe Power at lbd0');
        val = rep_nan (np.std (base_power[:,:,y0,b], axis=(0,1)));
        hdr[HMQ+'POWER'+name+' STD'] = (val,'Fringe Power at lbd0');

    # QC for bias
    log.info ('Compute QC for bias');
    
    qc_power = np.mean (bias_power[:,:,y0,:], axis=(0,1));
    hdr[HMQ+'BIASMEAN MEAN'] = (np.mean (qc_power),'Bias Power at lbd0');
    hdr[HMQ+'BIASMEAN STD'] = (np.std (qc_power),'Bias Power at lbd0');
    hdr[HMQ+'BIASMEAN MED'] = (np.median (qc_power),'Bias Power at lbd0');

