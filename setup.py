import numpy as np;
import os, ssl;

from scipy import interpolate 

import astropy;
from astropy.coordinates import EarthLocation, Angle, SkyCoord, ICRS, ITRS, AltAz;
from astropy import units;
from astropy.time import Time;

from .headers import HM, HMQ, HMP, HMW, HC, rep_nan;
from . import log;

# Default value for the IERS server
# astropy.utils.iers.conf.iers_auto_url = 'ftp://ftp.iers.org/products/eop/rapid/standard/finals2000A.data';
# astropy.utils.iers.conf.iers_auto_url = 'http://maia.usno.navy.mil/ser7/finals2000A.all';

# ensure astropy.coordinates can query the online database of locations:
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

def coef_flat(gain):
    gains = np.array([1, 10, 20, 50, 60]);
    coefs = np.array([-1.8883000e-06, -1.4006500e-06, -1.3885600e-06, -1.3524500e-06, -1.3416900e-06]);
    return interpolate.interp1d(gains, coefs)([gain])[0];

# Definition of setups
global target_names;
target_names = ['OBJECT']

global detwin;
detwin = ['CROPROWS','CROPCOLS'];

global detmode;
detmode = ['NREADS','NLOOPS','NBIN','GAIN','FRMPRST'];

global camtiming;
camtiming = ['NREADS','NLOOPS','NBIN','FRMPRST','CROPROWS','CROPCOLS']; #affects timing in detail

global insmode;
insmode = ['FILTER1','FILTER2','CONF_NA','MIRCX_SPECTRO_XY','MIRCX_SPECTRO_FOC','MIRC COMBINER_TYPE'];

global fringewin;
fringewin = [HMW+'FRINGE STARTX', HMW+'FRINGE NX', HMW+'FRINGE STARTY', HMW+'FRINGE NY'];

global visparam;
visparam = [HMP+'NCOHER'];

global beamorder;
beamorder = ['BEAMORD0','BEAMORD1','BEAMORD2','BEAMORD3','BEAMORD4','BEAMORD5'];

global pop;
pop = [HC+"S1_POP", HC+"S2_POP", HC+"E1_POP", HC+"E2_POP", HC+"W1_POP", HC+"W2_POP"];

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
    Return a guess of lbd0,deltaLbd depending on
    instrument setup where lbd0 is the central wavelength
    of spectrum and dlbd is a the bandpass of one channel.
    lbdref is the reference wavelength corresponding to
    the sampling returned by beam_freq.
    '''

    # For MYSTIC, we try to use the
    # user specified parameter
    if hdr['BEAMCOMB'] == 'MYSTIC' :
        lbdref = 1.93e-6;
        dlbd = - hdr['BANDWID'] * 1e-6 / (hdr['FR_ROW2'] - hdr['FR_ROW1']);
        lbd0 = hdr['WAVELEN'] * 1e-6;
        
    # MIRC configurations
    elif hdr['CONF_NA'] == 'H_PRISM':
        lbdref,lbd0,dlbd = 1.60736e-06, 1.60736e-06, 21.e-9;
    elif (hdr['CONF_NA'] == 'H_GRISM200'):
        lbdref,lbd0,dlbd = 1.60736e-06, 1.60736e-06, -8.2e-9;
    elif (hdr['CONF_NA'] == 'H_GRISM150'):
        lbdref,lbd0,dlbd = 1.60736e-06, 1.60736e-06, -8.2e-9;
    elif (hdr['CONF_NA'] == 'H_GRISM'):
        lbdref,lbd0,dlbd = 1.60736e-06, 1.60736e-06, -8.2e-9;
    elif (hdr['CONF_NA'] == 'H_GRISM200                    S1=0,S2=1,E1=2,E2=3,W1=4,W2=5'):
        lbdref,lbd0,dlbd = 1.60736e-06, 1.60736e-06, -8.2e-9;
        
    # temporary configurations. Not sure
    # the sign is correct
    elif hdr['CONF_NA'] == 'H_PRISM20' :
        lbdref,lbd0 = 1.60736e-06, 1.60736e-06;
        dlbd = lbd0 / 27.4;
    elif hdr['CONF_NA'] == 'H_PRISM40' :
        lbdref,lbd0 = 1.60736e-06, 1.60736e-06;
        dlbd = lbd0 / 49.2;
        
    # MIRCX configurations, J-band on top
    # of image except for the GRISM_190.
    elif hdr['CONF_NA'] == 'H_PRISM22' :
        lbdref,lbd0 = 1.60736e-06, 1.60736e-06;
        dlbd = -lbd0 / 22.;
    elif hdr['CONF_NA'] == 'H_PRISM50' :
        lbdref,lbd0 = 1.60736e-06, 1.60736e-06;
        dlbd = -lbd0 / 50.;
    elif hdr['CONF_NA'] == 'H_PRISM102' :
        lbdref,lbd0 = 1.60736e-06, 1.60736e-06;
        dlbd = -lbd0 / 102.;
    elif hdr['CONF_NA'] == 'H_GRISM190' :
        lbdref,lbd0 = 1.60736e-06, 1.60736e-06;
        dlbd = lbd0 / 190.0;

    # Unknown configuration
    else:
        log.error ('Unknown CONF_NA');
        raise ValueError('CONF_NA unsuported (yet?)');

    # Verbose
    log.info ('Configuration '+hdr['CONF_NA']+'lbd = %fum dlbd = %fum'%(lbd0*1e6,dlbd*1e6));

    return lbdref,lbd0,dlbd;

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
        pos = np.array([4,6,13,18,24,28]);
    # Fiber position in old MIRC
    else :
        pos = np.array([9,3,1,21,14,18]);
        
    return pos

def beam_freq (hdr):
    '''
    Return the fiber position in the v-groove
    in fringe/pixel at lbdref. The scale factor
    is given by the geometry of the combiner
    '''
    # Scaling in pix/fringe at highest spatial frequency
    # and for wavelength defined as lbd0

    # MYSTIC sampling and fiber position
    if hdr['BEAMCOMB'] == 'MYSTIC' :
        scale = 2.75;
        tmp = np.array([4,6,13,18,24,28]) * 1.0 - 1.0;
    
    # New MIRCX sampling and fiber position
    # 2018-11-20, the scale is multiplied by 1./1.014 to match
    # the absolute calibration by John on iotaPeg.
    elif ('P_ION' in hdr) == True :
        scale = 2.75147;
        tmp = np.array([4,6,13,18,24,28]) * 1.0 - 1.0;
    
    # Old MIRCX sampling and fiber position
    else :
        scale = 5.023;
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
    return np.array (['%i%i'%(t[0]+1,t[1]+1) for t in base_beam ()]);

def beam_name ():
    '''
    Return the beam name following convention 0-5
    '''
    return ['1','2','3','4','5','6'];

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
    return np.array (['%i%i%i'%(t[0]+1,t[1]+1,t[2]+1) for t in triplet_beam()]);

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

def normalize (v, axis=0):
    '''
    Return normalized vector, along one axis
    '''
    return v / np.linalg.norm (v, axis=axis);

def compute_uv_frame_basic (icrs, obstime):
    '''
    Return the reference frame of the uv plan of the object 'icrs'
    observed from CHARA at the obstime, expressed in the local
    observer frame in cartesian x,y,z (East, North, Up).

    uv_frame = get_uv_frame (icrs,obstime);

    uv_frame[0]: coordinates of unitary u in the local observer frame
    uv_frame[1]: coordinates of unitary v in the local observer frame
    '''
    
    # HA and DEC of object in ITRS
    try:
        astropy.utils.iers.conf.iers_auto_url = 'ftp://ftp.iers.org/products/eop/rapid/standard/finals2000A.data';
        itrs = icrs.transform_to (ITRS(obstime=obstime));
    except:
        log.warning ('Fail to run icrs.transform_to ITRS, so try with another server');
        astropy.utils.iers.conf.iers_auto_url = 'http://maia.usno.navy.mil/ser7/finals2000A.all';
        itrs = icrs.transform_to (ITRS(obstime=obstime));
        
    # CHARA site
    lon, lat = chara_coord (None);

    dec = itrs.spherical.lat;
    ha  = lon - itrs.spherical.lon;
    
    uv_frame = [[np.cos (ha.rad), -np.sin (ha.rad) * np.sin (lat.rad), np.sin (ha.rad) * np.cos (lat.rad)],
                [np.sin (dec.rad) * np.sin (ha.rad), (np.sin (dec.rad) * np.cos (ha.rad) * np.sin (lat.rad)  + np.cos (dec.rad) * np.cos (lat.rad)), (-np.sin (dec.rad) * np.cos (ha.rad) * np.cos (lat.rad)  + np.cos (dec.rad) * np.sin (lat.rad))]];
        
    return np.array (uv_frame);

def compute_uv_frame (icrs,obstime):
    '''
    Return the reference frame of the uv plan of the object 'icrs'
    observed from CHARA at the obstime, expressed in the local
    observer frame in cartesian x,y,z (East, North, Up).

    uv_frame = get_uv_frame (icrs,obstime);

    uv_frame[0]: coordinates of unitary u in the local observer frame
    uv_frame[1]: coordinates of unitary v in the local observer frame
    '''

    # Compute an asterism in the IRCS frame of small
    # offsets toward North (v) and East (u)
    delta = 10 / 3600.0 * np.pi / 180;

    rb = icrs.copy();
    rb_um = rb.directional_offset_by (90*units.deg,-delta*units.rad);
    rb_up = rb.directional_offset_by (90*units.deg, delta*units.rad);
    rb_vm = rb.directional_offset_by (0*units.deg, -delta*units.rad);
    rb_vp = rb.directional_offset_by (0*units.deg,  delta*units.rad);

    # Transforme this asterism to local, observer frame, using
    # the full IERS transformation. Output is expressed in
    # normalized cartesian coordinates (North,East,Up).
    try:
        astropy.utils.iers.conf.iers_auto_url = 'ftp://ftp.iers.org/products/eop/rapid/standard/finals2000A.data';
        aa = AltAz (obstime=obstime, location=EarthLocation.of_site('CHARA'));
    except:
        log.warning ('Fail to run icrs.transform_to ITRS, so try with another server');
        astropy.utils.iers.conf.iers_auto_url = 'http://maia.usno.navy.mil/ser7/finals2000A.all';
        aa = AltAz (obstime=obstime, location=EarthLocation.of_site('CHARA'));
    
    eWo    = normalize (rb.transform_to (aa).cartesian.get_xyz());
    eWo_um = normalize (rb_um.transform_to (aa).cartesian.get_xyz());
    eWo_up = normalize (rb_up.transform_to (aa).cartesian.get_xyz());
    eWo_vm = normalize (rb_vm.transform_to (aa).cartesian.get_xyz());
    eWo_vp = normalize (rb_vp.transform_to (aa).cartesian.get_xyz());

    # # Differentiate this asterism to construct the
    # # reference vector of v and u, in local frame
    # eUo  = (eWo_up - eWo_um);
    # eVo  = (eWo_vp - eWo_vm);
    # eUo  = eUo / np.linalg.norm (eUo, axis=0);
    # eVo  = eVo / np.linalg.norm (eVo, axis=0);
    
    # Compute the observed (eUo,eVo,eWo) reference frame
    eUo = np.cross (eWo_up, eWo_um, axis=0);
    eUo = np.cross (eWo, eUo, axis=0) / (2.*delta);
    eVo = np.cross (eWo_vp, eWo_vm, axis=0);
    eVo = np.cross (eWo, eVo, axis=0) / (2.*delta);

    # CHARA cartesian telescope position are defined
    # in East,North,Up while the cartersian position
    # of astropy are defined North,East,Up
    eUo  = eUo[[1,0,2]];
    eVo  = eVo[[1,0,2]];
    
    return np.array ([eUo,eVo]);

def tel_xyz (hdr):
    '''
    Return a dictionary with the telescope positions
    read from header, ez, nz, uz in [m]. If keys are
    not existing, they are created with default.
    '''
    
    # Default from 2010Jul20 (JDM)
    default = {};
    default['S1'] = [0.0,0.0,0.0];
    default['S2'] = [  -5.746854437,  33.580641636,    0.63671908];
    default['E1'] = [ 125.333989819, 305.932632737,  -5.909735735];
    default['W1'] = [-175.073332211, 216.320434499, -10.791111235];
    default['W2'] = [ -69.093582796, 199.334733235,   0.467336023];
    default['E2'] = [  70.396607118, 269.713272258,  -2.796743436];

    for t in default.keys():
        try:
            x = hdr['HIERARCH CHARA '+t+'_BASELINE_X'];
            y = hdr['HIERARCH CHARA '+t+'_BASELINE_Y'];
            z = hdr['HIERARCH CHARA '+t+'_BASELINE_Z'];
            default[t] = x,y,z;
        except:
            log.warning ('Cannot read XYZ of '+t+' (use default and set in header)');
            x,y,z = default[t];
            hdr['HIERARCH CHARA '+t+'_BASELINE_X'] = x;
            hdr['HIERARCH CHARA '+t+'_BASELINE_Y'] = y;
            hdr['HIERARCH CHARA '+t+'_BASELINE_Z'] = z;
            
    return default;

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

def sky_coord (hdr):
    '''
    Return the SkyCoord of the target in header,
    in the ICRS and at the mjd defined in header.
    '''
    
    # Read coordinate
    dec_icrs = Angle (hdr['DEC'], unit=units.deg);
    ra_icrs  = Angle (hdr['RA'], unit=units.hourangle);

    # Get distance from PARALLAX (sec)
    plx = hdr.get ('PARALLAX', 0.0);
    distance = 1./max(plx, 1e-6) * units.pc;

    try:
        # Read velocities
        pm_ra    = hdr['PM_RA'] * units.rad/units.yr;
        pm_dec   = hdr['PM_DEC'] * units.rad/units.yr;
        rad_vel  = 0.0 * units.km / units.s;

        # Build structure
        coord_icrs = SkyCoord (dec=dec_icrs,ra=ra_icrs,distance=distance,
                               radial_velocity=rad_vel,
                               pm_ra_cosdec=pm_ra,pm_dec=pm_dec,
                               obstime='J2000', frame="icrs");
        
        # Evolve at the time
        meantime    = Time (hdr['MJD-OBS'], format='mjd');
        coord_icrs = coord_icrs.apply_space_motion (new_obstime=meantime);
    
    except:
        log.info ('Cannot propagate PM_RA and PM_DEC in coordinates');
        
        # Build structure
        coord_icrs = SkyCoord (dec=dec_icrs,ra=ra_icrs,distance=distance, 
                               obstime='J2000', frame="icrs");

    return coord_icrs;
    
def base_uv (hdr):
    '''
    Return the uv coordinages of all 15 baselines
    ucoord[nbase],vcoord[nbase] read from HEADER
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

    return np.array ([u,v]);

def compute_base_uv (hdr,mjd=None,baseid='base'):
    '''
    Return the uv coordinages of the CHARA baselines
    ucoord[nbase],vcoord[nbase] at the time of observation.

    baseid='base' returns the uv-plan of the 15 baselines
    that is the UCOORD and VCOORD of OIFITS definition.
    baseid='base1' returns return the uv-plan of the first
    baselines of the 20 closures (U1COORD, U2COORD). Same
    for 'base2'.

    If given, the mdj parameter should match the number
    of computed baseline (either 15 or 20).
    '''
    log.info ('Compute uv with erfa');

    # Get the physical telescope position (read from header)
    telpos = tel_xyz (hdr);
    xyz = np.array ([telpos[t] for t in beam_tel (hdr)]);

    # Physical baseline
    if baseid == 'base':
        baseline = np.array ([xyz[t1,:]-xyz[t2,:] for t1,t2 in base_beam()]);
    elif baseid == 'base1':
        baseline = np.array ([xyz[t1,:]-xyz[t2,:] for t1,t2,t3 in triplet_beam()]);
    elif baseid == 'base2':
        baseline = np.array ([xyz[t2,:]-xyz[t3,:] for t1,t2,t3 in triplet_beam()]);
    else:
        raise ValueError ('baseid is not valid');

    # Default for time
    if mjd is None: mjd = np.ones (baseline.shape[0]) * hdr['MJD-OBS'];

    # Time as a valid Time object
    obstime = Time (mjd, format='mjd');
    
    # Object position in ICRS, at the
    # time of observation
    coord_icrs = sky_coord (hdr);

    # uv unitary directions expressed in the local frame
    uv_frame = compute_uv_frame (coord_icrs, obstime);

    # Set a keyword in header
    hdr.set ('HIERARCH MIRC PRO UV_EQUATION', 'ERFA', 'Type of computation for uv-plan');

    # Project telescope baseline into uv directions
    uv = np.einsum ('uxb,bx->ub', uv_frame, baseline);

    return uv;
    
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

def kappa (hdr):
    '''
    Return the expected value for the ration fringe_flux / xchan_flux
    '''
    if (hdr['MJD-OBS'] < 58362.00): return 40.0; # Old MIRC optics
    if (hdr['MJD-OBS'] < 58525.00): return 3.0;  # Thorlabs BS
    return 5.5;  # Polar optimised BS
    
def uv_maxrel_distance (u1,v1,u2,v2):
    '''
    Return max relative distance
    '''
    rel = np.sqrt ((u1-u2)**2 + (v1-v2)**2) / np.sqrt (u1**2 + u2**2 + 0.001);
    return rel.max();
