import numpy as np;

# Definition of setups
global detector
detector = ['NREADS','NLOOPS','NBIN','GAIN','CROPROWS','CROPCOLS','FRMPRST'];

global instrument
instrument = ['FILTER1','FILTER2','CONF_NA'];

def nspec (hdr):
    '''
    Return the expected number of spectral
    channel depending on the insturmental setup
    '''
    return 11;

def fringe_widthx (hdr):
    '''
    Return the expected size of the fringe in
    spatial direction depending on setup
    '''
    return 200;

def photo_widthx (hdr):
    '''
    Return the expected size of the fringe in
    spatial direction depending on setup
    '''
    return 5;

def lbd0 (hdr):
    '''
    Return lbd0,deltaLbd depending
    on instrument setup
    '''
    lbd0,dlbd = (1.65e-6,21.e-9);
    return lbd0,dlbd;
    
def beam_freq (hdr):
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
    tmp = np.array (['01','02','03','04','05',\
                     '12','13','14','15',\
                     '23','24','25',\
                     '34','35',\
                     '45']);
    return tmp;

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
            raise ValueError('Cannot load UV from header');

    # CHARA tel of the CHARA beams
    return u,v;
