import numpy as np;

from .headers import HM, HMQ, HMP, HMW, rep_nan;
from . import log;

# Definition of setups
global detwin;
detwin = ['CROPROWS','CROPCOLS'];

global detmode;
detmode = ['NREADS','NLOOPS','NBIN','GAIN','FRMPRST'];

global insmode;
insmode = ['FILTER1','FILTER2','CONF_NA'];

global fringewin;
fringewin = [HMW+'FRINGE STARTX', HMW+'FRINGE NX', HMW+'FRINGE STARTY', HMW+'FRINGE NY'];

global visparam;
visparam = [HMP+'NFRAME_COHER'];

global beamorder;
beamorder = ['BEAMORD0','BEAMORD1','BEAMORD2','BEAMORD3','BEAMORD4','BEAMORD5'];

def nspec (hdr):
    '''
    Return the expected number of spectral
    channel depending on the insturmental setup
    '''
    n = int((hdr['FR_ROW2'] - hdr['FR_ROW1'])/2)*2 - 1;
    log.info ('nrow = %i'%n);
    return n;
    # return 11;

def fringe_widthx (hdr):
    '''
    Return the expected size of the fringe in
    spatial direction depending on setup
    '''
    return 200;

def photo_widthx (hdr):
    '''
    Return the expected size of the xchan in
    spatial direction depending on setup
    '''
    return 5;

def lbd0 (hdr):
    '''
    Return lbd0,deltaLbd depending
    on instrument setup
    '''
    lbd0 = 1.60736e-06;

    if hdr['CONF_NA'] == 'H_PRISM':
        log.info ('H_PRISM setup');
        dlbd = 21.e-9;
    elif hdr['CONF_NA'] == 'H_GRISM200' or 'H_GRISM150':
        log.info ('H_GRISM setup');
        dlbd = 8.2e-9;
    else:
        log.warning ('Unknown spectral setup, assume low dispersion');
        dlbd = 21.e-9;
    
    if hdr['BANDWID'] < 0: dlbd = -dlbd;
    
    return lbd0,dlbd;
    
def beam_freq (hdr):
    '''
    Return the fiber position in the v-groove
    in fringe/pixel at lbd0. The scale factor
    is given by the geometry of the combiner:
    scale = lbd0 / (20 * 250e-6 / 0.375) / 24e-6
    '''
    # Scaling in pix/fringe at highest spatial frequency
    # and for wavelength defined as lbd0
    scale = 5.023;
    # Fiber position in v-groove
    tmp = np.array([9,3,1,21,14,18]) * 1.0 - 1.0;
    # Fiber position in fringe/pix
    tmp /= (tmp.max() - tmp.min()) * scale;
    return tmp;

def base_freq (hdr):
    '''
    Return the base frequency in fringe/pixel
    '''
    beams = base_beam ();
    freq  = beam_freq (hdr);
    return freq[beams[:,1]] - freq[beams[:,0]];
    
def base_beam ():
    '''
    Return the MIRC beam numbering for each base
    beam[15,2]
    '''
    tmp = np.array ([[0,1],[0,2],[0,3],[0,4],[0,5],\
                     [1,2],[1,3],[1,4],[1,5],\
                     [2,3],[2,4],[2,5],\
                     [3,4],[3,5],\
                     [4,5]]);
    return tmp;

def base_name ():
    '''
    Return the MIRC base name for each base
    name[15]
    '''
    return np.array (['%i%i'%(t[0],t[1]) for t in base_beam ()]);

''' beam to base matrix '''
beam_to_base = np.zeros ((15,6));
for b,beams in enumerate(base_beam ()):
    beam_to_base[b,beams[0]] = +1;
    beam_to_base[b,beams[1]] = -1;

def triplet_base ():
    '''
    Return the base of each triplet. The last base
    of the triplet is conjugated.
    '''
    tmp = np.array ([[0,5,1],[0,6,2],[0,7,3],[0,8,4], [1,9,2],[1,10,3],[1,11,4], [2,12,3],[2,13,4], [3,14,4],
                     [5,9,6],[5,10,7],[5,11,8], [6,12,7],[6,13,8], [7,14,8],
                     [9,12,10],[9,13,11], [10,14,11],
                     [12,14,13]]);
    return tmp;

def triplet_beam ():
    '''
    Return the beams of each triplet
    '''
    tmp = np.array ([[0,1,2],[0,1,3],[0,1,4],[0,1,5], [0,2,3],[0,2,4],[0,2,5], [0,3,4],[0,3,5], [0,4,5],
                     [1,2,3],[1,2,4],[1,2,5], [1,3,4],[1,3,5], [1,4,5],
                     [2,3,4],[2,3,5], [2,4,5],
                     [3,4,5]]);
    return tmp;

def triplet_name ():
    '''
    Return the MIRC triplet name for each triplet
    name[20]
    '''
    return np.array (['%i%i%i'%(t[0],t[1],t[2]) for t in triplet_beam()]);

def beam_tel (hdr):
    '''
    Return the telescope name of all 6 beams.
    tel[6]
    '''
    
    # CHARA beam of the MIRC beams
    cbeam = np.array ([hdr['BEAMORD%i'%i] for i in range(6)]);

    # Check configuration
    if hdr['TEL_KEY'] != 'S1=0,S2=1,E1=2,E2=3,W1=4,W2=5':
        raise ValueError('Configuration unsuported');
    else:
        ctel = np.array(['S1','S2','E1','E2','W1','W2']);

    # CHARA tel of the CHARA beams
    return ctel[cbeam];
    
def beam_index (hdr):
    '''
    Return the station index of all 6 beams.
    index[6]
    '''
    
    # CHARA beam of the MIRC beams
    cbeam = np.array ([hdr['BEAMORD%i'%i] for i in range(6)]);

    # Check configuration
    if hdr['TEL_KEY'] != 'S1=0,S2=1,E1=2,E2=3,W1=4,W2=5':
        raise ValueError('Configuration unsuported');
    else:
        cidx = np.array(range (1,7));

    # CHARA tel of the CHARA beams
    return cidx[cbeam];

def base_uv (hdr):
    '''
    Return the uv coordinages of all 15 baselines
    ucoord[nbase],vcoord[nbase]
    '''
    
    # Get the telescope names of each base
    tels = beam_tel (hdr)[base_beam ()];

    u = np.zeros (15);
    v = np.zeros (15);
    
    for b,t in enumerate(tels):
        if 'U_'+t[0]+'-'+t[1] in hdr:
            u[b] = hdr['U_'+t[0]+'-'+t[1]];
            v[b] = hdr['V_'+t[0]+'-'+t[1]];
        elif 'U_'+t[1]+'-'+t[0] in hdr:
            u[b] = -hdr['U_'+t[1]+'-'+t[0]];
            v[b] = -hdr['V_'+t[1]+'-'+t[0]];
        else:
            log.warning ('Cannot read UV base %i (%s-%s) in header.'%(b,t[0],t[1]));
            
        if u[b] == v[b]:
            log.warning ('ucoord == vcoord base %i (%s-%s) in header.'%(b,t[0],t[1]));

    # CHARA tel of the CHARA beams
    return u,v;
