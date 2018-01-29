import matplotlib.pyplot as plt;
import numpy as np;

from scipy.optimize import minimize_scalar, brentq

def myfunc (roc, fn1, n1, fn2, n2, x):
    r2 = x * fn1 * n1 / n2 / 2;
    roc = np.arcsin (r2/roc) + n2/n1 * (np.arcsin(r2/x) - np.arcsin(r2/roc)) - np.arcsin(fn2/2);
    return roc;

# Input parameters
fn1, n1, fn2, n2 = (1./2.5, 1.0, 1./10., 1.79);
d2 = 180.0; x = d2  * n2/n1 / fn1;

# Compute ROC
roc = brentq (myfunc, 100, 1000., args=(fn1, n1, fn2, n2, x));

# Print outputs
print ('fn1 = f/%.3f'%(1./fn1));
print ('n1  = %.3f'%(n1));
print ('fn2 = f/%.3f'%(1./fn2));
print ('n2  = %.3f'%(n2));
print ('diam  = %.3f um'%(d2));
print ('thick = %.3f um'%(x));
print ('roc   = %.3f um'%(roc));
print ('...');
print ('fn = %.3f'%(roc / (0.5 * d2)));

