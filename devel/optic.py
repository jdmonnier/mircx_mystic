import numpy as np;
import os;

class setup (list):
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
        for s in self:
            s.x *= xs;
            s.f *= xs;
            s.y *= ys;
            s.d *= ys;

    def move (self,xm=0,ym=0):
        for s in self:
            s.x += xm;
            s.y += ym;
            
    def recenter (self):
        x0 = np.mean ([s.x for s in self]);
        y0 = np.mean ([s.y for s in self]);
        for s in self:
            s.x -= x0 - 600;
            s.y -= y0 - 300;
    
class object:
    def __init__(self,x=0,f=100,d=100,y=0,n1=1.5,n2=1.55):
        self.x = x;
        self.y = y;
        self.f = f;
        self.d = d;
        self.n1 = n1;
        self.n2 = n2;

class lens(object):
    def tostr (self):
        out  = '{"type":"lens"';
        out += ',"p1":{"type":1,"x":%i,"y":%i,"exist":true}'%(self.x,self.y+self.d/2);
        out += ',"p2":{"type":1,"x":%i,"y":%i,"exist":true}'%(self.x,self.y-self.d/2);
        out += ',"p":%i}'%(self.f);
        return out;

class stop(object):
    def tostr(self):
        out  = '{"type":"blackline"';
        out += ',"p1":{"type":1,"x":%i,"y":%i,"exist":true}'%(self.x,self.y+self.d/2);
        out += ',"p2":{"type":1,"x":%i,"y":%i,"exist":true}'%(self.x,self.y+5*self.d/2);
        out += '},{"type":"blackline"';
        out += ',"p1":{"type":1,"x":%i,"y":%i,"exist":true}'%(self.x,self.y-self.d/2);
        out += ',"p2":{"type":1,"x":%i,"y":%i,"exist":true}'%(self.x,self.y-5*self.d/2);
        out += '}';
        return out;

class screen(object):
    def tostr(self):
        out  = '{"type":"blackline"';
        out += ',"p1":{"type":1,"x":%i,"y":%i,"exist":true}'%(self.x,self.y+self.d/2);
        out += ',"p2":{"type":1,"x":%i,"y":%i,"exist":true}'%(self.x,self.y-self.d/2);
        out += '}';
        return out;

class radian(object):
    def tostr(self):
        out = '{"type":"radiant","x":%i,"y":%i,"p":0.9}'%(self.x,self.y);
        return out;

class beam(object):
    def tostr(self):
        dx = self.d / self.f;
        out  = '{"type":"parallel",';
        out += '"p1":{"type":1,"x":%i,"y":%i,"exist":true},'%(self.x+dx/2,self.y+self.d/2);
        out += '"p2":{"type":1,"x":%i,"y":%i,"exist":true}'%(self.x-dx/2,self.y-self.d/2);
        out += ',"p":0.9}';
        return out;

class prism(object):
    def tostr(self):
        x = (self.x - self.d/2, self.x + self.d/2, self.x + self.d/2, self.x - self.d/2);
        y = (self.y - self.d/2, self.y - self.d/2, self.y + self.d/2, self.y + self.d/2);
        out  = '{"type":"refractor","path":[';
        out += '{"x":%i,"y":%i,"arc":false},'%(x[0],y[0]);
        out += '{"x":%i,"y":%i,"arc":false},'%(x[1],y[1]);
        out += '{"x":%i,"y":%i,"arc":false}'%(x[3],y[3]);
        out += '],"notDone":false,"p":%.4f},'%self.n1;
        out += '{"type":"refractor","path":[';
        out += '{"x":%i,"y":%i,"arc":false},'%(x[1],y[1]);
        out += '{"x":%i,"y":%i,"arc":false},'%(x[2],y[2]);
        out += '{"x":%i,"y":%i,"arc":false}'%(x[3],y[3]);
        out += '],"notDone":false,"p":%.4f}'%self.n2;
        return out;

def slit (x=0,d=10,fn=6):
    ''' Define a slit'''
    f0 = fn*d;
    x0 = x+f0;
    l1 = lens (x=x0,y=+d/2,d=d,f=f0);
    l2 = lens (x=x0,y=-d/2,d=d,f=f0);
    b = beam (x=x0+10,d=2.*d,f=np.inf);
    return [l1,l2,b];

def camera (xd=0.0, dd=7.68, xp=36.9, dp=9.5):
    ''' Define the CRED array and pupil '''
    return [screen (x=xd,d=dd), stop (x=xp,d=dp)];
