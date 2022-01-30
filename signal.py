import numpy as np;
import matplotlib.pyplot as plt;

import scipy;
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter;
from scipy.special import gammainc, gamma;
from scipy.interpolate import interp1d

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



def bootstrap_triangles_jdm (snr,gd):
    '''
    MIRC/JDM Method: Compute the best SNR and GD of each baseline when considering 
    also the boostraping capability of the array.
    snr and gd shall be of shape (...,nb)

    Return (snr_b, gd_b) of same size, but including bootstrap.
    '''

    log.info ('Bootstrap baselines with triangles using MIRC/JDM method');


    w=snr.copy()
    opd0=gd.copy()
    ns,nf,ny,nb=snr.shape
    a=np.zeros((ns,nf,ny,5,5))
    b=np.zeros((ns,nf,ny,5))
    gd_jdm = np.zeros((ns,nf,ny,15))


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
    
    OPD=opd0.copy()
    OPD=np.where(w <=1., 0.0, OPD)
    w=np.where(w <=1., .01, w)

    #inzero=np.argwhere(w <= 100.)
    #OPD[inzero]=0.0
    #w[inzero]=.01

    opd12=OPD[:,:,:,0];
    opd13=OPD[:,:,:,1];
    opd14=OPD[:,:,:,2];
    opd15=OPD[:,:,:,3];
    opd16=OPD[:,:,:,4];
    opd23=OPD[:,:,:,5];
    opd24=OPD[:,:,:,6];
    opd25=OPD[:,:,:,7];
    opd26=OPD[:,:,:,8];
    opd34=OPD[:,:,:,9];
    opd35=OPD[:,:,:,10];
    opd36=OPD[:,:,:,11];
    opd45=OPD[:,:,:,12];
    opd46=OPD[:,:,:,13];
    opd56=OPD[:,:,:,14];

    w12=w[:,:,:,0]+0.001;
    w13=w[:,:,:,1]+0.002;
    w14=w[:,:,:,2]+0.005;
    w15=w[:,:,:,3]+0.007;
    w16=w[:,:,:,4]+0.003;
    w23=w[:,:,:,5]+0.004;
    w24=w[:,:,:,6]+0.008;
    w25=w[:,:,:,7]+0.009;
    w26=w[:,:,:,8]+0.002;
    w34=w[:,:,:,9]+0.003;
    w35=w[:,:,:,10]+0.006;
    w36=w[:,:,:,11]+0.008;
    w45=w[:,:,:,12]+0.009;
    w46=w[:,:,:,13]+0.004;
    w56=w[:,:,:,14]+0.005;

    a[:,:,:,0,0] = w12+w23+w24+w25+w26;
    a[:,:,:,1,1] = w13+w23+w34+w35+w36;
    a[:,:,:,2,2] = w14+w24+w34+w45+w46;
    a[:,:,:,3,3] = w15+w25+w35+w45+w56;
    a[:,:,:,4,4] = w16+w26+w36+w46+w56;

    a[:,:,:,0,1] = -w23;
    a[:,:,:,0,2] = -w24;
    a[:,:,:,0,3] = -w25;
    a[:,:,:,0,4] = -w26;

    a[:,:,:,1,0] = -w23;
    a[:,:,:,1,2] = -w34;
    a[:,:,:,1,3] = -w35;
    a[:,:,:,1,4] = -w36;

    a[:,:,:,2,0] = -w24;
    a[:,:,:,2,1] = -w34;
    a[:,:,:,2,3] = -w45;
    a[:,:,:,2,4] = -w46;

    a[:,:,:,3,0] = -w25;
    a[:,:,:,3,1] = -w35;
    a[:,:,:,3,2] = -w45;
    a[:,:,:,3,4] = -w56;

    a[:,:,:,4,0] = -w26;
    a[:,:,:,4,1] = -w36;
    a[:,:,:,4,2] = -w46;
    a[:,:,:,4,3] = -w56;

    b[:,:,:,0] = w12*opd12 - w23*opd23 - w24*opd24 - w25*opd25 - w26*opd26;
    b[:,:,:,1] = w13*opd13 + w23*opd23 - w34*opd34 - w35*opd35 - w36*opd36;
    b[:,:,:,2] = w14*opd14 + w24*opd24 + w34*opd34 - w45*opd45 - w46*opd46;
    b[:,:,:,3] = w15*opd15 + w25*opd25 + w35*opd35 + w45*opd45 - w56*opd56;
    b[:,:,:,4] = w16*opd16 + w26*opd26 + w36*opd36 + w46*opd46 + w56*opd56;

    #invert!
    result=np.linalg.solve(a, b)

    gd_jdm[:,:,:,0]=result[:,:,:,0]
    gd_jdm[:,:,:,1]=result[:,:,:,1]
    gd_jdm[:,:,:,2]=result[:,:,:,2]
    gd_jdm[:,:,:,3]=result[:,:,:,3]
    gd_jdm[:,:,:,4]=result[:,:,:,4]
    gd_jdm[:,:,:,5]=result[:,:,:,1]-result[:,:,:,0]
    gd_jdm[:,:,:,6]=result[:,:,:,2]-result[:,:,:,0]
    gd_jdm[:,:,:,7]=result[:,:,:,3]-result[:,:,:,0]
    gd_jdm[:,:,:,8]=result[:,:,:,4]-result[:,:,:,0]
    gd_jdm[:,:,:,9]=result[:,:,:,2]-result[:,:,:,1]
    gd_jdm[:,:,:,10]=result[:,:,:,3]-result[:,:,:,1]
    gd_jdm[:,:,:,11]=result[:,:,:,4]-result[:,:,:,1]
    gd_jdm[:,:,:,12]=result[:,:,:,3]-result[:,:,:,2]
    gd_jdm[:,:,:,13]=result[:,:,:,4]-result[:,:,:,2]
    gd_jdm[:,:,:,14]=result[:,:,:,4]-result[:,:,:,3]

    return (snr_b,gd_jdm,result);


def gd_tracker(opds_trial,input_snr,gd_key):
    '''
    Used for fitting a self-consistent set of opds. input 5 telscope delays 
    and compare to the snr vectors in opds space.
    return a globabl metric base don logs of the snrs with thresholds.
    '''

    #log.info ('Bootstrap baselines with triangles using MIRC/JDM method');
    
    # probably replace as matrix in future for vectorizing.
    gd_jdm,snr_jdm =  get_gds(opds_trial,input_snr,gd_key)
    
    #fit_metric = np.sum(np.log10(snr_jdm))
    fit_metric = np.sum(snr_jdm)

    return (-fit_metric);

def get_gds(topds,input_snr,gd_key):
    '''
    Used for fitting a self-consistent set of opds. input 5 telscope delays 
    and compare to the snr vectors in opds space.
    return a gds and snrs for self-consistent set of delays.
    '''
    nscan,nb=input_snr.shape
    gd_jdm=np.zeros(nb)
    snr_jdm=np.zeros(nb)

    gd_jdm[0]=topds[0]
    gd_jdm[1]=topds[1]
    gd_jdm[2]=topds[2]
    gd_jdm[3]=topds[3]
    gd_jdm[4]=topds[4]
    gd_jdm[5]=topds[1]-topds[0]
    gd_jdm[6]=topds[2]-topds[0]
    gd_jdm[7]=topds[3]-topds[0]
    gd_jdm[8]=topds[4]-topds[0]
    gd_jdm[9]=topds[2]-topds[1]
    gd_jdm[10]=topds[3]-topds[1]
    gd_jdm[11]=topds[4]-topds[1]
    gd_jdm[12]=topds[3]-topds[2]
    gd_jdm[13]=topds[4]-topds[2]
    gd_jdm[14]=topds[4]-topds[3]

    # interpolate into the snr.
    for i in range(nb):
        #snr_func=interp1d(gd_key,input_snr[:,i],kind='cubic',bounds_error=False,fill_value=(input_snr[:,i]).min(),assume_sorted=True)
        snr_func=interp1d(gd_key,input_snr[:,i],kind='cubic',bounds_error=False,fill_value=1.,assume_sorted=True)

        snr_jdm[i]=snr_func(gd_jdm[i])
    
    return(gd_jdm,snr_jdm)


def get_gd_gravity(topds, bestsnr_snrs,bestsnr_indices,softlength=2.,nscan=None):
    '''
    Used for fitting a self-consistent set of opds. input 5 telscope delays 
    and compare to the snr vectors in opds space.
    return a gds and snrs for self-consistent set of delays.
    
    topds = (nramps,nframes, ntels=5)
    bestsnr_snrs = (nramps, nframes, npeaks, nbaselines )
    bestsnr_indices = (nramps, nframes, npeaks, nbaselines )  ; integers

    '''
    nr,nf,npeak,nt=topds.shape
    nr,nf,npeak,nb=bestsnr_snrs.shape
    OPL_TO_OPD = setup.beam_to_base;
    temp = setup.base_beam ()



    #photo_power = photo[:,:,:,setup.base_beam ()];
    #totflux = np.nansum(photo,axis=(1,3))
    #bp=np.nanmean(bias_power,axis=2)
    topds1= topds[:,:,:,setup.base_beam ()]
    gd_jdm= topds1[:,:,:,:,1] - topds1[:,:,:,:,0]
    # if gd_jdm > nscan/2 than wraparond. but.. does sign work in fordce equation.. will have to check.
    
    ##if nscan != None:
    #    gd_jdm= np.where( gd_jdm >nscan/2, gd_jdm-nscan  ,gd_jdm)
    #    gd_jdm= np.where( gd_jdm < -nscan/2, nscan + gd_jdm, gd_jdm)

    # alternatively instead of adding in a discontunity, we could copy the force centers +/- nscan and apply
    # global down-weight.
    if nscan != None:
        bestsnr_snrs=np.concatenate((bestsnr_snrs,bestsnr_snrs,bestsnr_snrs),axis=2)
        bestsnr_indices=np.concatenate((bestsnr_indices,bestsnr_indices+nscan,bestsnr_indices-nscan),axis=2)
        bestsnr_snrs = bestsnr_snrs*np.exp(-.5*((bestsnr_indices/(nscan/2.))**2))
    
    snr_wt = np.log10(np.maximum(bestsnr_snrs,1.0))
    #snr_wt = np.sqrt(bestsnr_snrs)

    gd_forces=np.empty( (nr,nf,1,0))
    gd_pot   =np.empty( (nr,nf,1,0))
    gd_offsets =gd_jdm-bestsnr_indices
    for i_b in range(nt):
        factor0=OPL_TO_OPD[:,i_b][None,None,None,:] 
        F0 =  np.sum(factor0*snr_wt *np.sign(gd_offsets)*softlength**2/ (gd_offsets**2+softlength**2) ,axis=(2,3))
        gd_forces =np.append(gd_forces,F0[:,:,None,None],axis=3)
        F1 = np.sum(-2*np.abs(factor0)*snr_wt *softlength/ np.sqrt(gd_offsets**2+softlength**2) ,axis=(2,3)) # approximate!
        gd_pot = np.append(gd_pot,F1[:,:,None,None],axis=3)        
  
    return(gd_forces,gd_pot,gd_jdm )


def topd_to_gds(topds):
    '''
    Used for fitting a self-consistent set of opds. input 5 telscope delays 
    and compare to the snr vectors in opds space.
    return a gds and snrs for self-consistent set of delays.
    
    topds = (nramps,nframes, ntels = 6)
    bestsnr_snrs = (nramps, nframes, npeaks, nbaselines )
    bestsnr_indices = (nramps, nframes, npeaks, nbaselines )  ; integers

    '''


    #photo_power = photo[:,:,:,setup.base_beam ()];
    #totflux = np.nansum(photo,axis=(1,3))
    #bp=np.nanmean(bias_power,axis=2)
    topds1= topds[:,:,:,setup.base_beam ()]
    gd_jdm= topds1[:,:,:,:,0] - topds1[:,:,:,:,1] 
    
  
    return(gd_jdm)

def psd_projection (scale, freq, freq0, delta0, data):
    '''
    Project the PSD into a scaled theoretical model,
    Return the merit function 1. - D.M / sqrt(D.D*M.M)
    '''

    # Scale the input frequencies
    freq_s = freq * scale;
    
    # Compute the model of  PSD
    model = np.sum (np.exp (- (freq_s[:,None] - freq0[None,:])**2 / delta0**2), axis=-1);

    if data is None:
        return model;

    # Return the merit function from the normalised projection
    weight = np.sqrt (np.sum (model * model) * np.sum (data * data));
    return 1. - np.sum (model*data) / weight;


def decoherence_free (x, vis2, cohtime, expo):
    '''
    Decoherence loss due to phase jitter, from Monnier equation:
    vis2*2.*cohtime/(expo*x) * ( igamma(1./expo,(x/cohtime)^(expo))*gamma(1./expo) -
                                (cohtime/x)*gamma(2./expo)*igamma(2./expo,(x/cohtime)^(expo)) )

    vis2 is the cohence without jitter, cohtime is the coherence time, expo is the exponent
    of the turbulent jitter (5/3 for Kolmogorof)
    '''
    xc  = x/cohtime;
    xce = (xc)**expo;
    y  = gammainc (1./expo, xce) * gamma (1./expo) - gamma (2./expo) / xc * gammainc (2./expo, xce);
    y *= 2. * vis2 / expo  / xc;
    return y;

def decoherence (x, vis2, cohtime):
    '''
    decoherence function with a fixed exponent
    '''
    expo = 1.5;
    xc  = x/cohtime;
    xce = (xc)**expo;
    y  = gammainc (1./expo, xce) * gamma (1./expo) - gamma (2./expo) / xc * gammainc (2./expo, xce);
    y *= 2. * vis2 / expo  / xc;
    return y;
