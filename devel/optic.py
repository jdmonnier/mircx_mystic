import numpy as np;
import os;

#
# Individual objects
#
    
class object:
    def __init__(self,x=0,f=100,d=100,y=0,n1=1.5,n2=1.55,fn=1.0):
        self.x = float(x);
        self.y = float(y);
        self.f = float(f);
        self.d = float(d);
        self.n1 = float(n1);
        self.n2 = float(n2);
        self.fn = float(fn);
        
    def __add__ (self,b):
        ''' Adding 2 objects create a setup '''
        c = setup([b]) if hasattr(b, '__iter__') is False else b;
        return setup ([self]) + b;

    def rescale (self,xs,ys):
        self.x *= xs;
        self.f *= xs;
        self.y *= ys;
        self.d *= ys;
        self.fn *= float(xs) / float(ys);
    
class lens(object):
    def tostr (self):
        out  = '{"type":"lens"';
        out += ',"p1":{"type":1,"x":%.3f,"y":%.3f,"exist":true}'%(self.x,self.y+self.d/2);
        out += ',"p2":{"type":1,"x":%.3f,"y":%.3f,"exist":true}'%(self.x,self.y-self.d/2);
        out += ',"p":%.3f}'%(self.f);
        return out;

class stop(object):
    def tostr(self):
        out  = '{"type":"blackline"';
        out += ',"p1":{"type":1,"x":%.3f,"y":%.3f,"exist":true}'%(self.x,self.y+self.d/2);
        out += ',"p2":{"type":1,"x":%.3f,"y":%.3f,"exist":true}'%(self.x,self.y+5*self.d/2);
        out += '},{"type":"blackline"';
        out += ',"p1":{"type":1,"x":%.3f,"y":%.3f,"exist":true}'%(self.x,self.y-self.d/2);
        out += ',"p2":{"type":1,"x":%.3f,"y":%.3f,"exist":true}'%(self.x,self.y-5*self.d/2);
        out += '}';
        return out;

class screen(object):
    def tostr(self):
        out  = '{"type":"blackline"';
        out += ',"p1":{"type":1,"x":%.3f,"y":%.3f,"exist":true}'%(self.x,self.y+self.d/2);
        out += ',"p2":{"type":1,"x":%.3f,"y":%.3f,"exist":true}'%(self.x,self.y-self.d/2);
        out += '}';
        return out;

class radian(object):
    def tostr(self):
        out = '{"type":"radiant","x":%.3f,"y":%.3f,"p":0.5}'%(self.x,self.y);
        return out;

class beam(object):
    def tostr(self):
        dx = self.d / self.fn;
        out  = '{"type":"parallel",';
        out += '"p1":{"type":1,"x":%.3f,"y":%.3f,"exist":true},'%(self.x+dx/2,self.y+self.d/2);
        out += '"p2":{"type":1,"x":%.3f,"y":%.3f,"exist":true}'%(self.x-dx/2,self.y-self.d/2);
        out += ',"p":0.5}';
        return out;

class diffracting(object):
    def tostr(self):
        fnall = self.fn*2 * 1./np.linspace (1,1./100,5);
        out1 = [beam (y=self.y,x=self.x,d=self.d,fn=fn).tostr() for fn in fnall];
        fnall = -self.fn*2 * 1./np.linspace (1,1./100,5);
        out2 = [beam (y=self.y,x=self.x,d=self.d,fn=fn).tostr() for fn in fnall];
        return ','.join(out1+out2);
    
class prism(object):
    '''
    self.d is the vertical size
    self.f is the horizontal size
    '''
    def tostr(self):
        x = (self.x - self.f/2, self.x + self.f/2, self.x + self.f/2, self.x - self.f/2);
        y = (self.y - self.d/2, self.y - self.d/2, self.y + self.d/2, self.y + self.d/2);
        out  = '{"type":"refractor","path":[';
        out += '{"x":%.3f,"y":%.3f,"arc":false},'%(x[0],y[0]);
        out += '{"x":%.3f,"y":%.3f,"arc":false},'%(x[1],y[1]);
        out += '{"x":%.3f,"y":%.3f,"arc":false}'%(x[3],y[3]);
        out += '],"notDone":false,"p":%.4f},'%self.n1;
        out += '{"type":"refractor","path":[';
        out += '{"x":%.3f,"y":%.3f,"arc":false},'%(x[1],y[1]);
        out += '{"x":%.3f,"y":%.3f,"arc":false},'%(x[2],y[2]);
        out += '{"x":%.3f,"y":%.3f,"arc":false}'%(x[3],y[3]);
        out += '],"notDone":false,"p":%.4f}'%self.n2;
        return out;

class splitter(object):
    def tostr(self):
        d = self.d;
        x = (self.x-d/2-1, self.x-d/2+1, self.x+d/2+1, self.x+d/2-1);
        d = np.abs (self.d);
        y = (self.y-d/2, self.y-d/2, self.y+d/2, self.y+d/2);
        out  = '{"type":"refractor","path":[';
        out += '{"x":%.3f,"y":%.3f,"arc":false},'%(x[0],y[0]);
        out += '{"x":%.3f,"y":%.3f,"arc":false},'%(x[1],y[1]);
        out += '{"x":%.3f,"y":%.3f,"arc":false},'%(x[2],y[2]);
        out += '{"x":%.3f,"y":%.3f,"arc":false}'%(x[3],y[3]);
        out += '],"notDone":false,"p":3.0}';
        return out;

    def rescale(self,xs,ys):
        if xs != ys: raise ValueError('Cannot rescale xs!=ys');
        object.rescale (self,xs=xs,ys=ys);

class mirror(object):
    def tostr(self):
        d = self.d;
        y = (self.y-d/2,self.y+d/2);
        d = self.d / self.fn;
        x = (self.x-d/2,self.x+d/2);
        out  = '{"type":"mirror",';
        out += '"p1":{"type":1,"x":%.3f,"y":%.3f,"exist":true},'%(x[0],y[0]);
        out += '"p2":{"type":1,"x":%.3f,"y":%.3f,"exist":true}}'%(x[1],y[1]);
        return out;

    def rescale(self,xs,ys):
        if xs != ys: raise ValueError('Cannot rescale xs!=ys');
        object.rescale (self,xs=xs,ys=ys);
    
#
# Complexe setup, made of several objects
#

class setup (list):

    def __iadd__ (self,b):
        c = setup([b]) if hasattr(b, '__iter__') is False else b;
        return setup (list(self) + c);
    
    def __add__ (self,b):
        c = setup([b]) if hasattr(b, '__iter__') is False else b;
        return setup (list(self) + c);
    
    def tostr (self,filename=None):
        '''
        Write the setup into a string and/or an ASCII file
        '''
        out  = "";
        out += '{"version":2,"objs":[';
        out += ','.join (s.tostr() for s in self);
        out += '],"mode":"light","rayDensity_light":7.0,"rayDensity_images":1,'
        out += '"observer":null,"origin":{"x":0,"y":0}}';
        
        if filename is not None:
            print ('Write '+filename);
            file = open (filename, "w");
            file.write (out);
            file.close ();
            
        return out;
    
    def rescale (self,xs=1.0,ys=1.0):
        for s in self: s.rescale (xs,ys);

    def move (self,xm=0,ym=0):
        for s in self:
            s.x += xm;
            s.y += ym;
            
    def recenter (self,xc=None,yc=None):
        if xc is None: xc = np.mean ([s.x for s in self]);
        if yc is None: yc = np.mean ([s.y for s in self]);
        for s in self:
            s.x -= xc - 500;
            s.y -= yc - 400;

def slit (x=0,d=10,fn=6):
    ''' Define a slit'''
    f0 = fn*d;
    x0 = x+f0;
    l1 = lens (x=x0,y=+d/2,d=d,f=f0);
    l2 = lens (x=x0,y=-d/2,d=d,f=f0);
    b = beam (x=x0+10,d=2.*d,fn=np.inf);
    return setup ([l1,l2,b]);

def camera (xd=0.0, dd=7.68, xp=36.9, dp=9.5):
    ''' Define the CRED array and pupil '''
    return setup ([screen (x=xd,d=dd), stop (x=xp,d=dp)]);
