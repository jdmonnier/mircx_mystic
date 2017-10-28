
import argparse

parser = argparse.ArgumentParser (description="", epilog="",
                                 formatter_class=argparse.RawDescriptionHelpFormatter);
TrueFalse = ['TRUE','FALSE'];
TrueFalseOverwrite = ['TRUE','FALSE','OVERWRITE'];

parser.add_argument ("--debug", dest="debug",default='FALSE',
                     choices=TrueFalse,
                     help="stop or error [TRUE]");

parser.add_argument ("--output-dir", dest="outputDir",default='./reduced/',
                     help="output directories for product");

parser.add_argument ("--max-file", dest="maxFile",default=300,type=int,
                     help="maximum nuber of file to load to build "
                          "product (speed-up for tests)");

parser.add_argument ("--delta-time", dest="delta",default=300,type=float,
                     help="maximum time between files to be groupped (s) [300]");

parser.add_argument ("--background", dest="background",default='FALSE',
                     choices=TrueFalseOverwrite,
                     help="compute the BACKGROUND products [FALSE]");

parser.add_argument ("--beam-map", dest="bmap",default='FALSE',
                     choices=TrueFalseOverwrite,
                     help="compute the BEAM_MAP products [FALSE]");

parser.add_argument ("--preproc", dest="preproc",default='FALSE',
                     choices=TrueFalseOverwrite,
                     help="compute the PREPROC products [FALSE]");
