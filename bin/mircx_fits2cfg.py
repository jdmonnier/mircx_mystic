#! /usr/bin/env python
# -*- coding: iso-8859-15 -*-

from astropy.io import fits
import argparse

parser = argparse.ArgumentParser(description='Add a single fits file')
parser.add_argument('file', metavar='file', help='Mirc-X/Mystic FITS File(s).')
args = parser.parse_args()

h = fits.open(args.file)[0].header

print("#Please note that white space in row direction is not allowed. ")
print("#Software thinks that line is the end of file.")
print("#detector gain.")
print("#crop cols -> detector coulmns: 1-10 implies 320 pixels, each channel 32 pixels ")
print("CROP_COLS", h["CROPCOLS"])
print("#crop rows -> detector subwindow rows. This should be subwindow1,subwindow2.")
print("CROP_ROWS", h["CROPROWS"])
print("#nreads in iota readout mode")
print("NREADS", h["NREADS"])
print("#nloops in iota readout mode")
print("NLOOPS", h["NLOOPS"])
print("#number of frames per reset in the detector readout")
print("NFRAMES_PER_RESET", h["FRMPRST"])
print("#fringe window coordinates")
print("FRINGE_COL1", h["FR_COL1"])
print("FRINGE_COL2", h["FR_COL2"])
print("FRINGE_ROW1", h["FR_ROW1"])
print("FRINGE_ROW2", h["FR_ROW2"])
print("#xchannel window coordinates")
print("XCHANNEL_COL1", h["XCH_COL1"])
print("XCHANNEL_COL2", h["XCH_COL2"])
print("XCHANNEL_ROW1", h["XCH_ROW1"])
print("XCHANNEL_ROW2", h["XCH_ROW2"])
print("# split between polarisation")
print("DELTAPOLROW", h["DPOL_ROW"])
print("#number of files for a fits file")
print("NFRAMES_PER_FILE", h["NAXIS3"] * h["NAXIS4"])
print("#coherent coadd integration before doing the fft")
print("COHERENT_COADD", h["NCOHER"])
print("#power spectrum coadds")
print("PS_COADD", h["PSCOADD"])
print("#number of bins of data to compress data if required")
print("NBIN", h["NBIN"])
print("#wavelength of operation")
print("LAMBDA", h["WAVELEN"])
print("#bandwidth")
print("BANDWIDTH", h["BANDWID"])
print("#Cpeak (15 values)")
CPEAK = "CPEAK "
for i in range(15):
    CPEAK += str(h["CPK_" + str(i)]) + " "
print(CPEAK)
print("#Cpeak0 (15 values)")
CPEAK0 = "CPEAK0 "
for i in range(15):
    CPEAK0 += str(h["CPK0_" + str(i)]) + " "
print(CPEAK0)
print("#Cpeak1 (15 values)")
CPEAK1 = "CPEAK0 "
for i in range(15):
    CPEAK1 += str(h["CPK1_" + str(i)]) + " "
print(CPEAK1)
print("#Frequency for noise")
print("BGNOISE", h["BKGNDNOI"])
print("#Filter1")
print("FILTER1_NAME", h["FILTER1"])
print("#Filter2")
print("FILTER2_NAME", h["FILTER2"])
print("#ConfigName")
print("CONFIGNAME", h["CONF_NA"])
print("#combinerType")
print("COMBINERTYPE", h["HIERARCH MIRC COMBINER_TYPE"])
print("#Xchan pos")
XCHAN = "XCHANPOS "
for i in range(6):
    XCHAN += str(h["HIERARCH MIRC XCHAN_POS" + str(i)]) + " "
print(XCHAN)
