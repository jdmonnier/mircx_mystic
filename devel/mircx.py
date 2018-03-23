from mircx_pipeline.devel import optic as opt;

import copy, os;
import numpy as np;

'''
This is the unfolded, perfect current MIRCx design.
The diffraction is implemented as an superposition
of input collimated beams.

Questions:

* The combiner is at 360mm from slit.
What is its focal length 330mm ??

'''

# Camera
relay = opt.camera ();

# objective, collimator, combiner with 1" optics
relay += opt.lens (x=100.,d=25,f=100.);
relay += opt.lens (x=275.,d=25,f=100.);
relay += opt.lens (x=275+100+200,d=25,f=200.); d0 = 375;
# relay += opt.lens (x=275.,d=25,f=50.);
# relay += opt.lens (x=275+50+100,d=25,f=100.); d0 = 325;

# We have 200, 250, 375

# Create spatial and spectral direction
spectral = copy.deepcopy (relay);
spatial  = copy.deepcopy (relay);

# Align the spectro
spatial[-2].y += 1.0;

# Add cylendrical lenght and prism in spectral direction
spectral += opt.lens (x=d0+30,d=12.5,f=30.);
spectral += opt.prism (x=220,d=25,f=25,n1=1.5,n2=1.51);

# Collimated beams in spectral direction
x = relay[-1].x + 100;
spectral += opt.diffracting (x=x,y=0,d=0.2,fn=100);

# Collimated beams in spatial direction
y = 0.25 * np.array ([0,3,24]);
for yi in (y - np.mean (y)):
    spatial += opt.diffracting (x=x,y=yi,d=0.2,fn=100);

# Add x-chan in the spatial direction
spatial += opt.splitter (x=d0+60, d=12.5, fn=1.0);
spatial += opt.mirror (x=d0+60, y=-25, d=12.5, fn=1.0);
spatial += opt.lens (x=d0+7,y=-25,d=12.5,f=32.);
spatial += opt.mirror (x=d0-15, y=-25, d=12.5, fn=-1.0);
spatial += opt.mirror (x=d0-15, y=+3, d=3.5, fn=-1.0);

# Move spectral
spectral.move (ym=+80);
    
# Rescale and recenter
full = spatial + spectral;
full.rescale (ys=10,xs=10);
full.recenter (xc=(375-20)*0,yc=0);

# Write
i = (i+1)%2 if 'i' in locals() else 0;
full.tostr (os.getenv('HOME')+'/mircx%i.lens'%i);

