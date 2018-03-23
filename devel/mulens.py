from mircx_pipeline.devel import optic as opt;

import copy, os;
import numpy as np;

# for fused silica use n= 1.4350 (2.2mu), fn=7.6
# for silicon use n=3.4303. (for 2.2 mu at 200K), ,fn=18


full  = opt.setup ();

# Pitch=180um, Spacing=0mm, Thick=1mm, ROC=900um, n1=3.43
# for y in [-180,0,180]:
#     full += opt.lens (y=y, x=-100, d=50,f=100);
#     full += opt.beam (y=y, x=-120, d=-50,fn=np.inf);
#     full += opt.mulens (y=y, d=180,f=1000,fn=10,n1=3.43);

# Pitch=180um, Spacing=0um, Thick=644um, ROC=358 to be confirmed, n1=1.79
for y in [-180,0,180]:
    full += opt.lens (y=y, x=-100, d=50,f=100);
    full += opt.beam (y=y, x=-120, d=-50,fn=np.inf);
    full += opt.mulens (y=y, d=180,f=644,fn=3.977, n1=1.79);

# Pitch=180um, Spacing=0um, Thick=500um, ROC=213um, n1=1.43
# for y in [-180,0,180]:
#     full += opt.lens (y=y, x=-100, d=50,f=100);
#     full += opt.beam (y=y, x=-120, d=-50,fn=np.inf);
#     full += opt.mulens (y=y, d=180,f=500,fn=2.3,n1=1.43);

# Pitch=250um, Spacing=100um, Thick=800um, ROC=303um, n1=1.43
# for y in [-250,0,250]:
#     d0 = 100./2.8;
#     full += opt.lens (y=y, x=-100, d=d0,f=100);
#     full += opt.beam (y=y, x=-120, d=-d0,fn=np.inf);
#     full += opt.mulens (y=y, x=100, d=250,f=800,fn=2.43,n1=1.43);


full += opt.ruler (x =500, y=250/2, d=50, fn=10);

# Rescale and recenter
full.rescale (ys=1,xs=1);
full.recenter (xc=0,yc=0);

# Write
i = (i+1)%2 if 'i' in locals() else 0;
full.tostr (os.getenv('HOME')+'/mulens%i.lens'%i);

print (full[-2].roc);


