import numpy as np;
import matplotlib.pyplot as plt;

from scipy.optimize import fsolve;

def dist (p1,p2):
    d = 0.0
    for i in np.arange(len(p1)): d += (p1[i] - p2[i])**2;
    return np.sqrt (d);

def myFunction (y):
    '''
    x is from fold to instrument
    y is from prism to opd-machine
    z is from table to top
    '''
    # h1,h2 = 6.0, 6.0; # Jacob
    h1,h2 = 4.25, 6.0; # Michigan
    # h1,h2 = 5.25, 7.55; # CHARA

    # positions of prism
    x1 = 3.125 - np.arange (6) * 1.25; # ok
    # y1 = np.array([0., 2.5, -3.0619, 3.0619, -2.5,0.]); # ok
    y1 = np.array([0., 2.34, -3.059, 3.06, -2.502, 0.]); # ok
    z1 = np.ones (6) * h1;
    
    # positions of opd-machine
    x2 = x1;
    y2 = y;
    z2 = np.ones (6) * h2;
    
    # position of fold
    x3 = x2;
    y3 = -np.arange (6) * 3.0;
    z3 = z2;

    # position of exits
    x4 = 3.125;
    y4 = y3;
    z4 = z3;

    # current OPL
    dd  = dist ((x1,y1,z1),(x2,y2,z2));
    dd += dist ((x2,y2,z2),(x3,y3,z3));
    dd += dist ((x3,y3,z3),(x4,y4,z4));

    # requested OPL (match Jacob)
    req =  33.902 + np.arange (6) * 11.2;

    # residuals
    return dd - req;

# Solve
y2guess = 20 + np.ones(6) * 3;
x2 = fsolve (myFunction, y2guess);

print (x2);

print (x2 - 15);
