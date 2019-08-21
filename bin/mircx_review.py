#! /usr/bin/env python
from __future__ import print_function
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import argparse
import matplotlib.pyplot as plt
from astropy.io import fits
from tqdm import tqdm

current = 0

def onkey(event):
    global current
    global fig
    global img
    global fname
    global typ
    global good
    global stat

    def myplot():
        fig.clf()
        if good[current]:
            stat = "Good"
        else:
            stat = "Bad"
        plt.imshow(img[current])
        plt.title(fname[current] + " [" + typ[current] + "] - " + stat)
        fig.canvas.draw()
    
    if event.key == 'right':
        if current >= len(fname)-1:
            return
        current = current + 1
        myplot()

    elif event.key == 'left':
        if current == 0:
            return
        current = current - 1
        myplot()
    
    elif event.key == 'b':
        good[current] = False
        myplot()

    elif event.key == 'g':
        good[current] = True
        myplot()

    elif event.key == 'q':
        plt.close(fig)
        bad = np.array([not g for g in good])
        bad_list = np.asarray(fname)[bad]
        if len(bad_list) == 0:
            print("No bad files")
        else:
            print("Bad files:")
            [print(b) for b in bad_list]

parser = argparse.ArgumentParser(description='[Left] and [Right] to scroll through images. [G] to mark good files. [B] to mark bad files. [Q] to quit.')
    
parser.add_argument('file', metavar='file', nargs='+', help='Mirc-X/Mystic FITS File(s).')
parser.add_argument ("--saturate", dest="saturate", type=int,
                     default=0, help="Saturate upper and lower bound of pixels");

args = parser.parse_args()

img = []
fname = []
typ = []
good = []

for f in tqdm(args.file):
    fname.append(f)
    with fits.open(f) as hdulist:
        if f[-7:] == 'fits.fz':
            # Manipulate the header to fake only one dimension
            nx = hdulist[1].header['NAXIS1']
            ny = hdulist[1].header['NAXIS2']
            nf = hdulist[1].header['NAXIS3']
            nr = hdulist[1].header['NAXIS4']
            hdulist[1].header['NAXIS'] = 1
            hdulist[1].header['NAXIS1'] = nr*nf*ny*nx
            # Uncompress and reshape data
            cube = hdulist[1].data
            cube.shape = (nr,nf,ny,nx)
            typ.append(hdulist[1].header["FILETYPE"])
        # Read normal data. 
        else:
            cube = hdulist[0].data
            typ.append(hdulist[0].header["FILETYPE"])
    image = np.sum(cube[:,-2,:,:] - cube[:,1,:,:], axis=0)
    flat = image.flatten()
    sort = np.argsort(flat)
    sat = round(len(flat) * args.saturate / 100)
    flat[sort[:sat]] = flat[sort[sat]]
    if sat == 0:
        sat = 1
    flat[sort[-sat:]] = flat[sort[-sat]]
    img.append(np.reshape(flat, np.shape(image)))
    good.append(True)

fig = plt.figure()
cid = fig.canvas.mpl_connect('key_press_event', onkey)
if good[current]:
    stat = "Good"
else:
    stat = "Bad"
plt.imshow(img[current])
plt.title(fname[current] + " [" + typ[current] + "] - " + stat)
plt.show()
