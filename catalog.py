from astropy.io import fits as pyfits;
import os;
import numpy as np;

import matplotlib.pyplot as plt;
import matplotlib.colors as mcolors;

from . import log, files, headers, signal;

# Try import astroquery
try:
    from astroquery.vizier import Vizier;
except:
    print ('WARNING: cannot import astroquery.vizier');
    print ('WARNING: some functionalities will crash');

# Columns of our generic catalog
columns = [('NAME','20A',''),
           ('ISCAL','I',''),
           ('RAJ2000','20A','hms'),
           ('DEJ2000','20A','dms'),
           ('_r','E','arcm'),
           ('SpType','20A',''),
           ('Vmag','E','mag'),
           ('Hmag','E','mag'),
           ('MODEL','20A',''),
           ('PARAM1','E',''),
           ('e_PARAM1','E',''),
           ('PARAM2','E',''),
           ('e_PARAM2','E',''),
           ('PARAM3','E',''),
           ('e_PARAM3','E',''),
           ('PARAM4','E',''),
           ('e_PARAM4','E',''),
           ('PARAM5','E',''),
           ('e_PARAM5','E','')];

def model (u, v, lbd, mjd, data):
    '''
    Models for calibration stars. u, v and lbd are in [m]
    mjd is in Modified Julian Day.
    u, v and lbd should be conformable.

    data should accept the following calls and return valid data:
    data['MODEL'], data['PARAM1'], data['e_PARAM1']...

    The function returns the a tupple with
    the complex vis and its error.

    '''
    name = data['MODEL'];
    
    if name == 'UDD':
        spf  = np.sqrt (u**2 + v**2) / lbd * 4.84813681109536e-09;
        diam  = data['PARAM1'];
        ediam = data['e_PARAM1'];
        vis  = signal.airy (diam * spf);
        evis = np.abs (signal.airy ((diam-ediam) * spf) - signal.airy ((diam+ediam) * spf));
        
    elif name == 'LDD':
        log.warning ('LDD model is crap !!!');
        vis = u + v;
        evis = vis * 0.0;
        
    else:
        raise ValueError ('Model name is unknown');
    
    return vis, evis;

def create_from_jsdc (filename, hdrs):
    '''
    Create a new catalog file for stars in hdrs
    by querying the JSDC.

    The hdrs argument can be a list of star name,
    or a list of headers loaded by the function
    headers.loaddir ();

    The function write the catalog as a FITS file
    called "filename.fits". It erase any file existing
    with the same name.
    '''

    # Import and init astroquery
    Vizier.columns = ['+_r','*'];

    # List of object
    objlist = list(set([h if type(h) is str else h['OBJECT'] for h in hdrs]));

    # Create catalog file
    hdu0 = pyfits.PrimaryHDU ([]);
    hdu0.header['FILETYPE'] = 'CATALOG';

    # Column with names and then other columns
    bincols = [pyfits.Column (name='NAME', format='20A', array=objlist)];

    # Add other columns
    for c in columns[1:]:
        bincols.append (pyfits.Column (name=c[0], format=c[1], unit=c[2]));

    # Create FITS binary table, empty except the names
    hdu1 = pyfits.BinTableHDU.from_columns (bincols);
    hdu1.header['EXTNAME'] = 'CATALOG';

    # Loop on object in the list
    for i,obj in enumerate (objlist):
        try:
            cat = Vizier.query_object (obj, catalog='JSDC')[0][0];
            log.info ('Find JSDC for '+obj);
            log.info ("diam = %.3f mas"%cat['UDDH']);
            log.info ("Hmag = %.3f mas"%cat['Hmag']);

            # Set all info available in catalog
            for c in columns:
                if c[0] in cat.colnames:
                    hdu1.data[i][c[0]] = cat[c[0]];

            # Set LDD model
            hdu1.data['MODEL'] = 'UDD';
            hdu1.data['PARAM1'] = cat['UDDH'];
            hdu1.data['e_PARAM1'] = cat['e_LDD'];

            # Check if confident to be a calibrator
            if cat['UDDH'] > 0 and cat['UDDH'] < 1.0 and cat['e_LDD'] < 0.3 and cat['_r'] < 1./10:
                log.info (obj+' declared as calibrator');
                hdu1.data['ISCAL'] = 1;
                
        except:
            log.info ('Cannot find JSDC for '+obj);

    # Remove file if existing
    if os.path.exists (filename):
        os.remove (filename);
            
    # Write file
    hdulist = pyfits.HDUList ([hdu0,hdu1]);
    files.write (hdulist, filename+'.fits');
    
