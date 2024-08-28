# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 20:31:08 2024

@author: lhoes
"""
import numpy as np
import matplotlib.pyplot as plt
from PlanetPositon import PP
from PlanetPositon import GP
d2r=np.pi/180
r2d=180/np.pi

def getangle(r1,r2):
    angle = np.arccos(np.dot(r1,np.array([1, 0, 0]))/(np.linalg.norm(r1)*np.linalg.norm(np.array([1, 0, 0]))))
    if r1[0]>=0 and r1[1]>=0:
        angle=angle
    elif r1[0]<0 and r1[1]>=0:
        angle=angle
    elif r1[0]<0 and r1[1]<0:
        angle=2*np.pi-angle
    elif r1[0]>=0 and r1[1]<0:
        angle=2*np.pi-angle
    angle2 = np.arccos(np.dot(r2,np.array([1, 0, 0]))/(np.linalg.norm(r2)*np.linalg.norm(np.array([1, 0, 0]))))
    if r2[0]>=0 and r2[1]>=0:
        angle2=angle2
    elif r2[0]<0 and r2[1]>=0:
        angle2=angle2
    elif r2[0]<0 and r2[1]<0:
        angle2=2*np.pi-angle2
    elif r2[0]>=0 and r2[1]<0:
        angle2=2*np.pi-angle2
    return angle2-angle

def R1(o):
    return np.array([[1,         0,          0],
                     [0, np.cos(o), -np.sin(o)],
                     [0, np.sin(o),  np.cos(o)]])

def R3(o):
    return np.array([[np.cos(o), -np.sin(o), 0],
                     [np.sin(o), np.cos(o),  0],
                     [        0,         0,  1]])
    
def c0(z):
    if z>0:
        return np.cos(z**(1/2))
    if z==0:
        return 1
    return np.cosh((-z)**(1/2))

def c1(z):
    if z>0:
        return np.sin(z**(1/2))/(z**(1/2))
    if z==0:
        return 1
    return np.sinh((-z)**(1/2))/((-z)**(1/2))
    
def c2(z):
    return (1-c0(z))/z

def c3(z):
    return (1-c1(z))/z

def c4(z):
    return (1/2-c2(z))/z

def c5(z):
    return (1/6-c3(z))/z


def fdt(dtheta,nrev=0,n=1001,r1=np.array([1, 0, 0]),r2=np.array([1, 0, 0])*2,mu=1) :
    rr1 = np.linalg.norm(r1)
    rr2 = np.linalg.norm(r2)
    A = np.sqrt(rr1)
    B = np.sqrt(rr2)*np.cos(dtheta/2)
    C = np.sqrt(rr2)*np.sin(dtheta/2)
    P = A**2+B**2+C**2
    Q = 2*A*B
    z=np.linspace(-(np.pi/2)**2,np.pi**2,n)
    dt=np.zeros((n,))
    for i in range(n):    
        Z=z[i]
        dtnrev=0
        if nrev>0:
            dtnrev = np.pi*nrev/(c1(Z)**3)*(1/(2*mu)*((P-Q*c0(Z))/Z)**3)**(1/2)
        dt[i] = (2*P*c3(4*Z)+Q*(c1(Z)*c2(4*Z)-2*c0(Z)*c3(4*Z)))/(c1(Z)**3)*(2*(P-Q*c0(Z))/mu)**(1/2)+dtnrev
    return dt
    
def OrbitType(dtheta,n=1001,r1=np.ones((3,)),r2=np.ones((3,))*2,mu=1):
    rr1 = np.linalg.norm(r1)
    rr2 = np.linalg.norm(r2)
    A = np.sqrt(rr1)
    B = np.sqrt(rr2)*np.cos(dtheta/2)
    C = np.sqrt(rr2)*np.sin(dtheta/2)
    P = A**2+B**2+C**2
    Q = 2*A*B
    z=np.linspace(-(np.pi/2)**2,np.pi**2,n)
    a=np.zeros((n,))
    for i in range(n):
        Z=z[i]
        a[i]=(P-Q*c0(Z))/(2*Z*c1(Z)**2)
    return a

def ffp(self,z):
    R=(2*self.P*c3(4*z)+self.Q*(c1(z)*c2(4*z)-2*c0(z)*c3(4*z)))
    S=c1(z)**3
    V=self.P-self.Q*c0(z)
    T=(2*V/self.mu)**(1/2)
    U=0
    if self.nrev>0:
        U=np.pi*self.nrev/(c1(z)**3)*(1/(2*self.mu)*((self.P-self.Q*c0(z))/z)**3)**(1/2)
    dt=R/S*T+U
    dR=4*self.P*(3*c5(4*z)-c4(4*z))+self.Q*(1/2*c2(4*z)*(c3(z)-c2(z))+2*c1(z)*(2*c4(4*z)-c3(4*z))+c3(4*z)*c1(z)-4*c0(z)*(3*c5(4*z)-c4(4*z)))
    dS=3/2*c1(z)**2*(c3(z)-c2(z))
    dT=self.Q/((8*self.mu*V)**(1/2))*c1(z)
    dU=0
    if self.nrev>0:
        dU=-3/2*(np.pi*self.nrev)/(z**2*c1(z)**4)*(V/(2*z*self.mu))**1/2*(-self.Q/2*z*c1(z)**2+V*(c1(z)+z*(c3(z)-c2(z))))
    ddt=T/S*dR-R*T/(S**2)*dS+R/S*dT+dU
    
    return dt,ddt


def rotate(r1,dtheta):
    return np.array([r1[0]*np.cos(dtheta)-np.sin(dtheta)*r1[1], r1[0]*np.sin(dtheta)+r1[1]*np.cos(dtheta), 0])

class Lambert:
    def __init__(self,dtheta,dt,nrev,LoS,tm,mu,r1,r2):
        self.dtheta=dtheta%(2*np.pi)
        if tm==-1:
            self.dtheta=2*np.pi-dtheta
        self.dt = dt
        self.nrev = nrev
        self.LoS = LoS
        self.tm = tm
        self.mu = mu
        self.r1 = r1
        self.r2 = r2
        self.ogangle = getangle(np.array([1, 0, 0]),r1)
        self.rr1 = np.linalg.norm(r1)
        self.rr2 = np.linalg.norm(r2)
        self.A = np.sqrt(self.rr1)
        self.B = np.sqrt(self.rr2)*np.cos(dtheta/2)
        self.C = np.sqrt(self.rr2)*np.sin(dtheta/2)
        self.P = self.A**2+self.B**2+self.C**2
        self.Q = 2*self.A*self.B
        self.n=1001
        return
    
    def __repr__(self):
        for attr in dir(self):
            if attr[0]!='_':
                print("{} = {}".format(attr,getattr(self,attr)))
        return ' '
    
    def __str__(self):
        plt.figure()
        plt.arrow(0,0,self.r1[0],self.r1[1],width=0.05,length_includes_head=True,head_length=0.1,color="blue")
        plt.arrow(0,0,self.r2[0],self.r2[1],width=0.05,length_includes_head=True,head_length=0.1,color="red")
        if self.tm==-1:
            angle=np.arange(self.O1+self.ogangle,self.ogangle+self.O1-self.dtheta+self.tm*np.pi/180,self.tm*np.pi/180)
        elif self.tm==1:
            OO=self.O2
            if self.O2<self.O1:
                OO=self.O2+2*np.pi
            angle=np.arange(self.O1+self.ogangle,self.ogangle+OO+self.tm*np.pi/180,self.tm*np.pi/180)
            
        courbe=np.array([self.p/(1+self.e*np.cos(angle))*np.cos(angle-self.O1),
                          self.p/(1+self.e*np.cos(angle))*np.sin(angle-self.O1)])
        self.angle=angle
        self.courbe=courbe
        thisvalue=np.max(np.abs((self.r1,self.r2)))
        othervalue=np.max(np.abs(courbe))
        thisvalue=np.nanmax((thisvalue,othervalue))+1
        plt.xlim(-thisvalue,thisvalue)
        plt.ylim(-thisvalue,thisvalue)
        plt.plot(courbe[0],courbe[1],'g')
        plt.grid()
            
        plt.figure()
        z=np.linspace(-(np.pi/2)**2,np.pi**2,self.n)
        dt=np.zeros((self.n,))
        for i in range(self.n):    
            Z=z[i]
            dtnrev=0
            if self.nrev>0:
                dtnrev = np.pi*self.nrev/(c1(Z)**3)*(1/(2*self.mu)*((self.P-self.Q*c0(Z))/Z)**3)**(1/2)
            dt[i] = (2*self.P*c3(4*Z)+self.Q*(c1(Z)*c2(4*Z)-2*c0(Z)*c3(4*Z)))/(c1(Z)**3)*(2*(self.P-self.Q*c0(Z))/self.mu)**(1/2)+dtnrev
        plt.plot(z,dt,label=str(round(self.dtheta*180/np.pi)))
        
        plt.arrow(-(np.pi/2)**2,self.realdt,self.z+(np.pi/2)**2,0,width=0.05,length_includes_head=True,head_length=0.1,color="orange",label="dt="+str(round(self.realdt,2)))
        plt.arrow(self.z,0,0,self.realdt,width=0.05,length_includes_head=True,head_length=0.1,color="brown",label="z="+str(round(self.z,2)))
        plt.xlim(-(np.pi/2)**2,np.pi**2)
        plt.ylim(0,50)
        plt.legend()
        plt.xlabel("z (NU)")
        plt.ylabel("dt (NU)")
        plt.grid()
        plt.show()
        return ' '
        
    
    def _getdtmin(self):
        z=np.linspace(-(np.pi/2)**2,np.pi**2,self.n)
        dt=np.zeros((self.n,))
        for i in range(self.n):    
            Z=z[i]
            dtnrev=0
            if self.nrev>0:
                dtnrev = np.pi*self.nrev/(c1(Z)**3)*(1/(2*self.mu)*((self.P-self.Q*c0(Z))/Z)**3)**(1/2)
            dt[i] = (2*self.P*c3(4*Z)+self.Q*(c1(Z)*c2(4*Z)-2*c0(Z)*c3(4*Z)))/(c1(Z)**3)*(2*(self.P-self.Q*c0(Z))/self.mu)**(1/2)+dtnrev
        self.dtmin=np.nanmin(dt)
        self.zmin=z[list(dt).index(np.nanmin(dt))]
        self.dtbefore=dt
        if self.nrev>0:
            if self.LoS==0:
                condition = z<self.zmin
            else : 
                condition = z>self.zmin
            self.condition= condition
            dt[np.invert(condition)] = dt[np.invert(condition)]+self.dt
            idx = np.nanargmin((np.abs(dt-self.dt)))
            self.idx=idx
            self.bigdt=dt
            self.dtclose = dt[idx]
            self.zclose = z[idx]
            return
        # self.bigdt = dt
        idx = np.nanargmin((np.abs(dt-self.dt)))
        self.dtclose = dt[idx]
        self.zclose = z[idx]
        return
    
    def _getaep(self,z):
        a=(self.P-self.Q*c0(z))/(2*z*c1(z)**2)
        self.a=a
        self.e=(1-(self.A**2*self.C**2)/(a**2*z*c1(z)**2))**(1/2)
        self.p=2*self.A**2*self.C**2/(self.P-self.Q*c0(z))
        return
    
    def _getz(self):
        n=0
        max_iter=100
        precision=1e-15
        z0=self.z0
        while n<max_iter :
            f,fp=ffp(self,z0)
            n+=1
            z1=z0+(self.dt-f)/fp
            if np.abs(z1-z0)<precision :
                break
            z0=z1
        self.niter=n
        self.z=z1
        return
    
    def _getv(self):
        self.O1=np.arccos(((self.p/self.rr1-1)/self.e))
        self.O1=self.O1%(2*np.pi)
        if self.dtheta>=3*np.pi/2 and self.rr1<self.rr2 and self.tm==1:
            self.O1=np.arccos(((self.p/self.rr1-1)/self.e))
            self.O1=2*np.pi-self.O1
        self.O2=self.O1+self.dtheta
        self.O2=self.O2%(2*np.pi)
        # print(self.O2==np.arccos(((self.p/self.rr2-1)/self.e)))
        # print(self.O2,np.arccos(((self.p/self.rr2-1)/self.e)))
        self.E1=2*np.arctan(((1-self.e)/(1+self.e))**(1/2)*np.arctan(self.O1/2))
        self.E2=2*np.arctan(((1-self.e)/(1+self.e))**(1/2)*np.arctan(self.O2/2))
        self.M1=self.E1-self.e*np.sin(self.E1)
        self.M2=self.E2-self.e*np.sin(self.E2)
        T=2*np.pi*(self.a**3/self.mu)**(1/2)
        self.dM=2*np.pi/T*self.dt
        self.testdt=self.dM*T/(2*np.pi)
        # if round(self.testdt,3)!=round(self.realdt,3):
        #     print("realdt vs testdt :",self.realdt,self.testdt)
        self.v1=(self.mu/self.p)**(1/2)*np.array([self.e*np.sin(self.O1), 1+self.e*np.cos(self.O1), 0])
        self.v2=(self.mu/self.p)**(1/2)*np.array([self.e*np.sin(self.O2), 1+self.e*np.cos(self.O2), 0])
    
    def _print_3d(self,Earth,Planet,YY,MM,DD,hh,mm,ss):
        ite=1000
        PositionEarth=np.zeros((ite,3))
        PositionPlanet=np.zeros((ite,3))
        ran=np.linspace(0,1,ite)
        n=0
        for i in ran:
            xE,xS,vE,vS=PP(YY=YY+i,MM=MM,DD=DD,hh=hh,mm=mm,ss=ss,EarthE=Earth,SaturnE=Planet,dt=0)
            PositionEarth[n]=xE
            PositionPlanet[n]=xS
            n+=1
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot([0,self.r1[0]],[0,self.r1[1]],[0,self.r1[2]],color='blue')
        ax.plot([0,self.r2[0]],[0,self.r2[1]],[0,self.r2[2]],color='red')
        ax.plot(PositionEarth[:,0],PositionEarth[:,1],PositionEarth[:,2],color="blue",alpha=0.3)
        ax.plot(PositionPlanet[:,0],PositionPlanet[:,1],PositionPlanet[:,2],color='red',alpha=0.3)
        
        # letheta=self.O1
        # if self.tm==1:
        #     angle=np.arange(letheta,(self.dtheta+np.pi/180+letheta),np.pi/180)
        # else :
        #     angle=np.arange(-letheta,(self.dtheta-2*np.pi-np.pi/180)-letheta,-np.pi/180)
        # self.angle=angle
        # r=self.a*(1-self.e**2)/(1+self.e*np.cos(angle))
        # self.r=r
        # courbe=np.array([r*np.cos(angle-self.tm*letheta), r*np.sin(angle-self.tm*letheta)])
        if self.tm==-1:
            angle=np.arange(self.O1+self.ogangle,self.ogangle+self.O1-self.dtheta+self.tm*np.pi/180,self.tm*np.pi/180)
        elif self.tm==1:
            OO=self.O2
            if self.O2<self.O1:
                OO=self.O2+2*np.pi
            angle=np.arange(self.O1+self.ogangle,self.ogangle+OO+self.tm*np.pi/180,self.tm*np.pi/180)
            
        courbe=np.array([self.p/(1+self.e*np.cos(angle))*np.cos(angle-self.O1),
                          self.p/(1+self.e*np.cos(angle))*np.sin(angle-self.O1)])
        self.r=self.a*(1-self.e**2)/(1+self.e*np.cos(angle))
        self.angle=angle
        self.courbe=courbe
        #3d attempt
        # self.h=self.r1*self.r2
        # self.inc=np.arccos(self.h[2]/np.linalg.norm(self.h))
        # self.N=np.array([-self.h[1], self.h[0], 0])
        # if self.N[1]>=0:
        #         self.OMEGA=np.arccos(self.N[0]/np.linalg.norm(self.N))
        # else:
        #     self.OMEGA=2*np.pi - np.arccos(self.N[0]/np.linalg.norm(self.N))
        
        # self.u1=np.arccos((self.r1[0]*np.cos(self.OMEGA)+self.r1[2]*np.sin(self.OMEGA))/np.linalg.norm(self.r1))
        # if self.r1[2]>=0:
        #     self.omega=self.u1-self.O1
        # else :
        #     self.omega=-self.u1-self.O1
        # courbecourbe = np.vstack((self.courbe,np.zeros((1,(self.courbe).shape[1]))))
        # courbecourbe=R3(-self.OMEGA)@R1(-self.inc)@R3(-self.omega)@courbecourbe
        # self.cc=courbecourbe
        
        thisvalue=np.max(np.abs((self.r1,self.r2)))
        othervalue=np.max(np.abs(self.r))
        thisvalue=np.nanmax((thisvalue,othervalue))+1
        self.thisvalue=thisvalue
        plt.xlim(-thisvalue,thisvalue)
        plt.ylim(-thisvalue,thisvalue)
        plt.plot(courbe[0],courbe[1],color='green')
        #ax.plot(courbecourbe[0],courbecourbe[1],courbecourbe[2],color='yellow')
        
        ax.set_xlim(-thisvalue,thisvalue)
        ax.set_ylim(-thisvalue,thisvalue)
        ax.set_zlim(-0.1,0.1)
        return PositionEarth