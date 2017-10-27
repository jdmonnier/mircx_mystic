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
