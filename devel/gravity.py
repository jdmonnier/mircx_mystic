from mircx_pipeline.devel import optic as opt;

import copy, os;
import numpy as np;

'''
This is the unfolted, perfect current MIRCx design.
The diffraction is implemented as an superposition
of input collimated beams.

Questions:

* The combiner is at 360mm from slit.
What is its focal length 330mm ??

'''

# Camera
relay = opt.camera ();

# objective, collimator, combiner
relay += opt.lens (x=100.,d=45,f=100.);
relay += opt.lens (x=275.,d=45,f=100.);
# We have 200, 250, 375

# Create spatial and spectral direction
spectral = copy.deepcopy (relay);
spatial  = copy.deepcopy (relay);

# Add cylendrical lenght and prism in spectral direction
spectral += opt.prism (x=220,d=45,f=45,n1=1.5,n2=1.51);

# Combiner in spectral direction
x = relay[-1].x + 100;
spectral += opt.diffracting (x=x,y=0,d=1.0,fn=4);

# Combiner in spatial direction
spatial += opt.diffracting (x=x,y=0,d=4,fn=4);

# for yi in (y - np.mean (y)):
#     spatial += opt.diffracting (x=x,y=yi,d=0.1,fn=2.5);

# Move spectral
spectral.move (ym=+60);
    
# Rescale and recenter
full = spatial + spectral;
full.rescale (ys=5,xs=5);
full.recenter (xc=None,yc=None);

# Write
i = (i+1)%2 if 'i' in locals() else 0;
full.tostr (os.getenv('HOME')+'/gravity%i.lens'%i);

