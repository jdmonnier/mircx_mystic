import numpy as np;
import os;

import astropy;
from astropy.coordinates import EarthLocation, Angle;
from astropy import units;
from astropy.time import Time;

from .headers import HM, HMQ, HMP, HMW, HC, rep_nan;
from . import log;

# Default value for the IERS server
astropy.utils.iers.conf.iers_auto_url = 'ftp://ftp.iers.org/products/eop/rapid/standard/finals2000A.data';
# astropy.utils.iers.conf.iers_auto_url = 'http://maia.usno.navy.mil/ser7/finals2000A.all';

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
visparam = [HMP+'NCOHER'];

global beamorder;
beamorder = ['BEAMORD0','BEAMORD1','BEAMORD2','BEAMORD3','BEAMORD4','BEAMORD5'];

global pop;
pop = [HC+"S1_POP", HC+"S2_POP", HC+"E1_POP", HC+"E2_POP", HC+"W1_POP", HC+"W2_POP"]

# Directory for static calibration
global static;
static = os.path.dirname (os.path.abspath(__file__))+'/static/';

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

    # MIRC configurations
    if hdr['CONF_NA'] == 'H_PRISM':
        dlbd = 21.e-9;
    elif (hdr['CONF_NA'] == 'H_GRISM200'):
        dlbd = -8.2e-9;
    elif (hdr['CONF_NA'] == 'H_GRISM150'):
        dlbd = -8.2e-9;
        
    # temporary configurations. Not sure
    # the sign is correct
    elif hdr['CONF_NA'] == 'H_PRISM20' :
        dlbd = lbd0 / 27.4;
    elif hdr['CONF_NA'] == 'H_PRISM40' :
        dlbd = lbd0 / 49.2;
        
    # MIRCX configurations, J-band on top
    # of image except for the GRISM_190.
    elif hdr['CONF_NA'] == 'H_PRISM22' :
        dlbd = -lbd0 / 22.;
    elif hdr['CONF_NA'] == 'H_PRISM50' :
        dlbd = -lbd0 / 50.;
    elif hdr['CONF_NA'] == 'H_PRISM102' :
        dlbd = -lbd0 / 102.;
    elif hdr['CONF_NA'] == 'H_GRISM190' :
        dlbd = lbd0 / 190.0;

    # Unknown configuration
    else:
        log.error ('Unknown CONF_NA');
        raise ValueError('CONF_NA unsuported (yet?)');

    # Verbose
    log.info ('Configuration '+hdr['CONF_NA']+' dlbd = %fum'%(dlbd*1e6));

    return lbd0,dlbd;

def xchan_ratio(hdr):
    '''
    Return a crude estimate
    '''
    if ('P_ION' in hdr) == True :
        return 0.3;
    else:
        return 0.1;

def fiber_pos(hdr):
    '''
    Return the fiber position in the v-groove
    in unit of micro-lenses
    '''
    
    # Fiber position in new MIRC-X
    if ('P_ION' in hdr) == True :
        pos = np.array([4,6,13,18,24,28])
    # Fiber position in old MIRC
    else :
        pos = np.array([9,3,1,21,14,18])
        
    return pos

def beam_freq (hdr):
    '''
    Return the fiber position in the v-groove
    in fringe/pixel at lbd0. The scale factor
    is given by the geometry of the combiner:
    scale = lbd0 / (20 * 250e-6 / 0.375) / 24e-6
    '''
    # Scaling in pix/fringe at highest spatial frequency
    # and for wavelength defined as lbd0
    # 2018-11-20, the scale is multiplied by 1./1.014 to match
    # the absolute calibration by John on iotaPeg.
    
    # Check if it's old MIRC or new MIRC-X data
    if ('P_ION' in hdr) == True :
        # scale = 2.78999;
        scale = 2.75147; 
        # Fiber position in v-groove
        tmp = np.array([4,6,13,18,24,28]) * 1.0 - 1.0;
        # Fiber position in fringe/pix
        tmp /= (tmp.max() - tmp.min()) * scale;
    
    else :
        scale = 5.023;
        # Fiber position in v-groove
        tmp = np.array([9,3,1,21,14,18]) * 1.0 - 1.0;
        # Fiber position in fringe/pix
        tmp /= (tmp.max() - tmp.min()) * scale;
    return tmp;

def ifreq_max (hdr):
    '''
    Return the highest frequency to use, as integer number
    '''
    if ('P_ION' in hdr) == True :
        return 72;
    else:
        return 40;

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

def beam_name ():
    '''
    Return the beam name following convention 0-5
    '''
    return ['0','1','2','3','4','5'];

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
    if ('P_ION' in hdr) == True :
        ctel = np.array(['S1','S2','E1','E2','W1','W2']);
    
    else :
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
    if ('P_ION' in hdr) == True :
        cidx = np.array(range (1,7));
    
    else :
        if hdr['TEL_KEY'] != 'S1=0,S2=1,E1=2,E2=3,W1=4,W2=5':
            raise ValueError('Configuration unsuported');
        else:
            cidx = np.array(range (1,7));

    # CHARA tel of the CHARA beams
    return cidx[cbeam];

def beam_xyz (hdr):
    '''
    Return a dictionary with the telescope positions
    read from header, ez, nz, uz in [m]. The output
    is of shape (6,3). The beams are ordered for MIRCX.
    '''

    # Default from 2010Jul20 (JDM)
    default = {};
    default['S1'] = [0.0,0.0,0.0];
    default['S2'] = [  -5.746854437,  33.580641636,    0.63671908];
    default['E1'] = [ 125.333989819, 305.932632737,  -5.909735735];
    default['W1'] = [-175.073332211, 216.320434499, -10.791111235];
    default['W2'] = [ -69.093582796, 199.334733235,   0.467336023];
    default['E2'] = [  70.396607118, 269.713272258,  -2.796743436];

    # Get the telescope names of each base
    pos = 0.0 * np.zeros ((6,3));
    for i,t in enumerate (beam_tel (hdr)):
        try:
            x = hdr['HIERARCH CHARA '+t+'_BASELINE_X'];
            y = hdr['HIERARCH CHARA '+t+'_BASELINE_Y'];
            z = hdr['HIERARCH CHARA '+t+'_BASELINE_Z'];
            pos[i,:] = x,y,z;
        except:
            log.warning ('Cannot read XYZ of '+t+' (use default)');
            pos[i,:] = default[t];

    return pos;

def chara_coord (hdr):
    '''
    Return the longitude and latitude of CHARA
    '''
    # lon = Angle (-118.059166, unit=units.deg);
    # lat = Angle (34.231666, unit=units.deg);
    
    c = EarthLocation.of_site ('CHARA');
    if hasattr (c, 'lon'):
        return c.lon, c.lat;
    else:
        return c.longitude, c.latitude;
    
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
    return np.array ([u,v]);

def compute_base_uv (hdr,mjd=None):
    '''
    Return the uv coordinages of all 15 baselines
    ucoord[nbase],vcoord[nbase]
    '''
    log.info ('Compute uv');

    # Default for time
    if mjd is None: mjd = np.ones (15) * hdr['MJD-OBS'];
    obstime = Time (mjd, format='mjd');

    # Get the physical baseline (read from header)
    xyz = beam_xyz (hdr);
    baseline = np.array ([xyz[t1,:] - xyz[t2,:] for t1,t2 in base_beam()]);

    # CHARA site
    lon, lat = chara_coord (hdr);
    
    # HA and DEC
    dec = Angle (hdr['DEC'], unit=units.deg);
    ra  = Angle (hdr['RA'], unit=units.hourangle);
    ha  = obstime.sidereal_time ('apparent', longitude=lon) - ra;
    
    # Project baseline on sky
    bx = -np.sin (lat.rad) * baseline[:,1] + np.cos (lat.rad) * baseline[:,2];
    by = baseline[:,0]
    bz = np.cos (lat.rad) * baseline[:,1] + np.sin (lat.rad) * baseline[:,2];

    # Now convert bx,by,bz to (u,v,w)
    u =  np.sin (ha.rad) * bx + np.cos (ha.rad) * by;
    v = -np.sin (dec.rad) * np.cos (ha.rad) * bx + np.sin (dec.rad) * np.sin (ha.rad) * by + np.cos (dec.rad) * bz;

    return np.array ([u,v]);

def crop_ids (hdr):
    '''
    Read the cropping parameter of the HDR
    and return the ids to crop a full-frame
    '''

    croprows = hdr['CROPROWS'].split(',');
    cropcols = hdr['CROPCOLS'];

    # Spatial direction
    if cropcols != '1-10':
        raise ValueError ('CROPCOLS of 1-10 supported only');
    else:
        idx = np.arange (0, 320);

    # Spectral direction
    idy = np.array([], dtype=int);
    for win in croprows:
        a,b = win.split('-');
        idy = np.append (idy, np.arange (int(a), int(b)+1));

    return idy,idx;
 
