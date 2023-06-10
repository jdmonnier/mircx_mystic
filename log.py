from timeit import default_timer as timer
import time, sys, os, logging, psutil;
import traceback;

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

# Create the logger
logger = logging.getLogger ('mircx_mystic_pipeline');
logger.setLevel (logging.INFO);

# Create the handler for stream
logStream = logging.StreamHandler();
logger.addHandler (logStream);

# Set the formater for this handler
formatter = logging.Formatter (
     "[%(color)s%(levelname)-7.7s"+RESET+"] %(asctime)s.%(msecs)03d [%(memory)s]: %(message)s",
     datefmt='%Y-%m-%dT%H:%M:%S');
logStream.setFormatter (formatter);
logStream.setLevel (logging.INFO);

def setFile (filename):
    '''
    Set a log file. The file is ensured
    to be writable by all group.
    '''
    for h in logger.handlers:
        if type(h) == logging.FileHandler:
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
                                   "%(asctime)s.%(msecs)03d [%(memory)s]: %(message)s",
                                    datefmt='%Y-%m-%dT%H:%M:%S');
    logfile.setFormatter (formatter);
    logger.addHandler (logfile);

def closeFile ():
    '''
    Stop logging in files
    '''
    for h in logger.handlers:
        if type(h) == logging.FileHandler:
            logger.removeHandler (h);

def memory ():
    '''
    Get memory usage of the process, in
    human readble string
    '''
    value = psutil.Process(os.getpid()).memory_info().rss;
    if value >= 1e8: return '%5.2fG'%(value/1e9);
    if value >= 1e5: return '%5.2fM'%(value/1e6);
    return '%5.2fk'%(value/1e3);

# Logging functions
def info(msg):
    mem = memory ();
    logger.info (msg, extra={'color':BLUE,'memory':mem});

def warning(msg):
    mem = memory ();
    logger.warning (msg, extra={'color':MAGENTA,'memory':mem});

def check(flag,msg):
    mem = memory ();
    if flag:
        logger.warning (msg, extra={'color':MAGENTA,'memory':mem});
    else:
        logger.info (msg, extra={'color':BLUE,'memory':mem});

def error(msg):
    mem = memory ();
    logger.error (traceback.format_exc(), extra={'color':RED,'memory':mem});
    logger.error (msg, extra={'color':RED,'memory':mem});

def debug(msg):
    mem = memory ();
    logger.debug (debug, extra={'color':RED,'memory':mem});
    
# Trace class (measure time until killed)
class trace:
    def __init__(self, funcname,color=True):
        self.color = GREEN if color else BLUE;
        self.funcname = funcname;
        mem = memory ();
        logger.info('Start '+funcname,extra={'color':self.color,'memory':mem});
        self.stime = timer();

    def __del__(self):
        if self.stime is not None and self.funcname is not None:
            mem = memory ();
            msg = 'End '+self.funcname+' in %.2fs'%(timer()-self.stime);
            logger.info (msg,extra={'color':self.color,'memory':mem});
        
