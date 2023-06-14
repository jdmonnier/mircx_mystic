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
    COL_WARNING = col.Fore.YELLOW
    COL_ERROR   = col.Fore.RED 
    COL_DEBUG   = col.Fore.BLUE
    COL_INFO    = col.Fore.GREEN
    COL_CRITICAL= col.Fore.WHITE + col.Back.BLACK
    RED     = col.Fore.RED;
    MAGENTA = col.Fore.MAGENTA;
    RESET   = col.Fore.RESET + col.Back.RESET;
    BLUE    = col.Fore.BLUE;
    GREEN   = col.Fore.GREEN;
    YELLOW  = col.Fore.YELLOW;

#Fore: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
#Back: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
#Style: DIM, NORMAL, BRIGHT, RESET_ALL

# Create the logger
logger = logging.getLogger ('mircx_mystic_pipeline');
logger.setLevel (logging.DEBUG); # show everything that comess through. adjust level through stream

# Create the handler for stream
logStream = logging.StreamHandler();
logger.addHandler (logStream);

# Set the formater for this handler
formatter = logging.Formatter (
     "[%(color)s%(levelname)-8.8s"+RESET+"] %(asctime)s.%(msecs)03d [%(memory)s]: %(message)s",
     datefmt='%Y-%m-%dT%H:%M:%S');
logStream.setFormatter (formatter);
logStream.setLevel (logging.DEBUG); 

def setFile (filename):
    '''
    Set a log file. The file is ensured
    to be writable by all group.
    '''
    for h in logger.handlers:
        if type(h) == logging.FileHandler:
            logger.removeHandler (h);

    # Create logfile and set permission
    open (filename, 'w').close();
    os.chmod (filename,0o666);

    # Set this file as log (mode 'append')
    # since file already exists
    logfile = logging.FileHandler (filename, mode='a');
    logfile.setLevel (logging.DEBUG);
    # No color in the file....
    formatter = logging.Formatter ("[%(levelname)-8.8s] "
                                   "%(asctime)s.%(msecs)03d [%(memory)s]: %(message)s",
                                    datefmt='%Y-%m-%dT%H:%M:%S');
    logfile.setFormatter (formatter);
    logger.addHandler (logfile);
    info ('Set logFile: '+filename);


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

def setLevel(logLevel):
    logger.setLevel(getattr(logging, logLevel))

# Logging functions


def info(msg):
    mem = memory ();
    logger.info (msg, extra={'color':COL_INFO,'memory':mem});

def warning(msg):
    mem = memory ();
    logger.warning (msg, extra={'color':COL_WARNING,'memory':mem});

def check(flag,msg):
    mem = memory ();
    if flag:
        logger.warning (msg, extra={'color':COL_WARNING,'memory':mem});
    else:
        logger.info (msg, extra={'color':COL_INFO,'memory':mem});

def error(msg):
    mem = memory ();
    logger.error (traceback.format_exc(), extra={'color':COL_ERROR,'memory':mem});
    logger.error (msg, extra={'color':COL_ERROR,'memory':mem});

def debug(msg):
    mem = memory ();
    logger.debug (msg, extra={'color':COL_DEBUG,'memory':mem});
    
def critical(msg):
    mem = memory ();
    logger.critical (msg, extra={'color':COL_CRITICAL,'memory':mem});

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
        
