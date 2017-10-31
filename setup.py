import numpy as np;

# Definition of setups
global detector
detector = ['NREADS','NLOOPS','NBIN','GAIN','CROPROWS','CROPCOLS','FRMPRST'];

global instrument
instrument = ['FILTER1','FILTER2','CONF_NA'];

def get_nspec (hdr):
    '''
    Return the expected number of spectral
    channel depending on the insturmental setup
    '''
    return 11;

def get_fringe_widthx (hdr):
    '''
    Return the expected size of the fringe in
    spatial direction depending on setup
    '''
    return 200;

def get_photo_widthx (hdr):
    '''
    Return the expected size of the fringe in
    spatial direction depending on setup
    '''
    return 5;

def get_lbd0 (hdr):
    '''
    Return lbd0,deltaLbd depending
    on instrument setup
    '''
    lbd0,dlbd = (1.65e-6,21.e-9);
    return lbd0,dlbd;
    
def get_beam_freq (hdr):
    '''
    Return the fiber position in the v-groove
    in fringe/pixel at lbd0
    '''
    # Scaling in pix/fringe at highest spatial frequency
    # and for central wavelength defined as lbd0
    scale = 5.023;
    # Fiber position in v-groove
    tmp = np.array([9,3,1,21,14,18]) * 1.0 - 1.0;
    # Fiber position in fringe/pix
    tmp /= (tmp.max() - tmp.min()) * scale;
    return tmp;

def get_base_freq (hdr):
    '''
    Return the base frequency in fringe/pixel
    '''
    beams = get_base_beam ();
    freq  = get_beam_freq (hdr);
    return freq[beams[:,1]] - freq[beams[:,0]];
    
def get_base_beam ():
    '''
    Return the beam numbering for each base
    '''
    tmp = np.array ([[0,1],[0,2],[0,3],[0,4],[0,5],\
                     [1,2],[1,3],[1,4],[1,5],\
                     [2,3],[2,4],[2,5],\
                     [3,4],[3,5],\
                     [4,5]]);
    return tmp;

def get_base_name ():
    '''
    Return the base name for each base
    '''
    tmp = np.array (['01','02','03','04','05',\
                     '12','13','14','15',\
                     '23','24','25',\
                     '34','35',\
                     '45']);
    return tmp;
