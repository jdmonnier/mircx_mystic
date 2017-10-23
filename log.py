from timeit import default_timer as timer
import time
import sys
import logging

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


# Setup the configuration to log in a file
logging.basicConfig(
     level=logging.DEBUG,
     format="[%(levelname)-7.7s] %(asctime)s: %(message)s",
     datefmt='%Y-%m-%dT%H:%M:%S',
     filename='mircx_pipeline.log', filemode='w');

# Get the logger
logger = logging.getLogger('mircx_pipeline');

# Also log in consode, with colors
console = logging.StreamHandler();
console.setLevel (logging.INFO);
formatter = logging.Formatter("[%(color)s%(levelname)-7.7s"+RESET+"] "
                              "%(asctime)s: %(message)s",
                              datefmt='%Y-%m-%dT%H:%M:%S');
console.setFormatter (formatter);
logger.addHandler(console);

# Logging function
def info(msg):
    logger.info (msg, extra={'color':BLUE});

def warning(msg):
    logger.warning (msg, extra={'color':MAGENTA});

def error(msg):
    logger.error (msg, extra={'color':RED});

class trace:
    def __init__(self, funcname):
        logger.info('Start '+funcname,extra={'color':GREEN});
        self.funcname = funcname;
        self.stime = timer();

    def __del__(self):
        if self.stime is not None and self.funcname is not None:
            msg = 'End '+self.funcname+' in %.2fs'%(timer()-self.stime);
            logger.info (msg,extra={'color':GREEN});
        
