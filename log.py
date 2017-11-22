from timeit import default_timer as timer
import time, sys, os, logging, psutil;

# Load colors
try:
    import colorama as col
except:
    RED     = '';
    MAGENTA = '';
    RESET   = '';
    BLUE    = '';
    GREEN   = '';
else:
    RED     = col.Fore.RED;
    MAGENTA = col.Fore.MAGENTA;
    RESET   = col.Fore.RESET;
    BLUE    = col.Fore.BLUE;
    GREEN   = col.Fore.GREEN;

# Get the logger
logger = logging.getLogger ('mircx_pipeline');

# Setup the configuration to log in the consol
logging.basicConfig (
     level=logging.INFO,
     # format="[%(color)s%(levelname)-7.7s"+RESET+"] %(asctime)s.%(msecs)03d: %(message)s",
     format="[%(color)s%(levelname)-7.7s"+RESET+"] %(asctime)s.%(msecs)03d [%(memory)s]: %(message)s",
     datefmt='%Y-%m-%dT%H:%M:%S');

# Set a logfile
def setFile (filename):
    for h in logger.handlers:
        logger.removeHandler (h);

    # Create logfile and set permission
    info ('Set logFile: '+filename);
    open (filename, 'w').close();
    os.chmod (filename,0o666);

    # Set this file as log (mode 'append')
    # since file already exists
    logfile = logging.FileHandler (filename, mode='a');
    logfile.setLevel (logging.INFO);
    formatter = logging.Formatter ("[%(levelname)-7.7s] "
                                   "%(asctime)s.%(msecs)03d: %(message)s",
                                    datefmt='%Y-%m-%dT%H:%M:%S');
    logfile.setFormatter (formatter);
    logger.addHandler (logfile);

# Stop logging in files
def closeFile ():
    for h in logger.handlers:
        logger.removeHandler (h);

# Get memory usage in usefull string
def memory ():
    value = psutil.Process(os.getpid()).memory_info().rss;
    if value >= 1e8: return '%5.2fG'%(value/1e9);
    if value >= 1e5: return '%5.2fM'%(value/1e6);
    return '%5.2fk'%(value/1e3);

# Logging function
def info(msg):
    mem = memory ();
    logger.info (msg, extra={'color':BLUE,'memory':mem});

def warning(msg):
    mem = memory ();
    logger.warning (msg, extra={'color':MAGENTA,'memory':mem});

def error(msg):
    mem = memory ();
    logger.error (msg, extra={'color':RED,'memory':mem});

def debug(msg):
    mem = memory ();
    logger.debug (debug, extra={'color':RESET,'memory':mem});
    
# Trace class (measure time until killed)
class trace:
    def __init__(self, funcname):
        mem = memory ();
        logger.info('Start '+funcname,extra={'color':GREEN,'memory':mem});
        self.funcname = funcname;
        self.stime = timer();

    def __del__(self):
        if self.stime is not None and self.funcname is not None:
            mem = memory ();
            msg = 'End '+self.funcname+' in %.2fs'%(timer()-self.stime);
            logger.info (msg,extra={'color':GREEN,'memory':mem});
        
