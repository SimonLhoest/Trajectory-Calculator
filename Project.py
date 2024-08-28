# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 20:32:02 2024

@author: lhoes
"""
from Lambert import *
from PlanetPositon import PP
from PlanetPositon import GP

def Project(dtheta, dt,r1,r2,tm=1,nrev=0,LoS=0,mu=1):
    Object = Lambert(dtheta,dt,nrev,LoS,tm ,mu,r1,r2)
    Object._getdtmin()
    if dt<Object.dtmin:
        #print("The problem does not admit solution : dt<dtmin \n")
        return Object
    Object._getaep(Object.zclose)
    if Object.a>0:
        Object.OT='Ellipse'
    else:
        Object.OT='Hyperbola'
    Object.z0=Object.zclose
    Object._getz()
    Object.realdt=ffp(Object,Object.z)[0]
    if np.abs(ffp(Object,Object.zclose)[0]-dt)<np.abs(Object.realdt-dt):
        Object.realdt=ffp(Object,Object.zclose)[0]
        Object.z=Object.zclose
    Object._getaep(Object.z)
    Object._getv()
    return Object

#%% R1 ET R2 CLAAASSIQUE
dtheta=2*np.pi/2
r1=np.array([1, 0, 0])
r2=0.01*rotate(r1,dtheta)
tm=1
LoS=0
O=Project(dtheta,2,r1,r2,tm=tm,nrev=0,LoS=LoS)
print(O)
self=O
#%% PLANET 1 PLAT NORMALISE VECTEUR 
YY=2020
MM=9
DD=10
hh=8
mm=56
ss=31
dt=75
#a, e, i, Omega, omega, L
#AU, , °, °, °, °
bigdi=GP()
EarthE=bigdi["Earth"]
SaturnE=bigdi["Mars"]
#ECC
EarthE[:,1]=0
SaturnE[:,1]=0
#INC
EarthE[:,2]=0
SaturnE[:,2]=0

xEn,xSn,vE,vS=PP(YY,MM,DD,hh,mm,ss,EarthE,SaturnE,dt)

tm=1
vangle=getangle(xEn,xSn)
OGangle=getangle(np.array([1, 0, 0]),xEn)

r1=[np.linalg.norm(xEn),0,0]
r2=rotate(np.array([1, 0]),vangle)*np.linalg.norm(xSn)
 
O2=Project(vangle,dt,r1,r2,tm)
print(O2)
self=O2
#%% PLANET 2 PLAT NON NORMAL
dtheta=vangle
r1=xEn
r2=xSn
O1=Project(dtheta,dt,r1,r2,tm)
print(O1)
self=O1
    
#%% 3D TEST 
#a, e, i, Omega, omega, L
#AU, , °, °, °, °
bigdi=GP()
EarthE=bigdi["Earth"]
SaturnE=bigdi["Mars"]
xE,xS,vE,vS=PP(YY,MM,DD,hh,mm,ss,EarthE,SaturnE,dt)
tm=1
dtheta=getangle(xE,xS)
r1=xE
r1=np.array([np.linalg.norm(r1[0:2]), 0, r1[2]])
r2=xS
r2=np.linalg.norm(r2[0:2])*rotate(np.array([1, 0]),dtheta)+np.array([0, 0, r2[2]])

O3=Project(dtheta,dt,r1,r2,tm)
O3._print_3d(EarthE,SaturnE,YY,MM,DD,hh,mm,ss)
self=O3
    
#%% TEST PCP
YY=2020
MM=1
DD=1
hh=0
mm=0
ss=1
import time as t
#a, e, i, Omega, omega, L
#AU, , °, °, °, °
bigdi=GP()
EarthE=bigdi["Earth"]
SaturnE=bigdi["Mercury"]
G = 6.67*10**(-11)
muS = G*1.989*10**(30) #km3/s2
AU2km = 149597870.7
km2AU = 1/AU2km
RealmuS = muS*km2AU**3
start=t.time()
n=185
dV=np.zeros((n,n))*np.nan
dt=np.linspace(1,750,n)
date=np.linspace(1,1827,n)
for i in range(n):
    print(i)
    for j in range(n):
        YY=2020
        MM=1
        DD=1+date[j]
        hh=0
        mm=0
        ss=1
        xE,xS,vE,vS=PP(YY,MM,DD,hh,mm,ss,EarthE,SaturnE,dt[i])
        dtheta=getangle(xE,xS)
        tm=-1
        r1=xE
        r2=xS
        O2=Project(dtheta=dtheta, dt=dt[i], nrev=0,LoS=0,tm=tm,mu=1,r1=r1,r2=r2)
        try:
            dV[j,i]=np.linalg.norm(np.abs(O2.v1-vE*km2AU)+np.abs(vS*km2AU-O2.v2))
        except AttributeError:  
            continue
stop=t.time()
print('Exec time :',stop-start)
#%%SHOW PCP data
print(np.where(dV==np.nanmin(dV)))
print(np.nanmin(dV))
print("days : ",date[np.where(dV==np.nanmin(dV))[0][0]])
print("dt : ",dt[np.where(dV==np.nanmin(dV))[1][0]])
#date, dt
#%% Display PCP
plt.figure()
plt.title("dV")
plt.imshow(np.flipud(dV), cmap = 'plasma',  extent=[0, 10, 0, 10])
plt.colorbar(label='dV')
plt.ylabel("Departure years from 01/01/2020")
plt.yticks(np.arange(11), labels = np.linspace(0,5,11))
plt.xlabel("Time of flight (days)")
plt.xticks(np.arange(0,11,2.5), labels = np.linspace(0,750,5))
plt.show()
        

#%% Find dV for submit date
YY=2024
MM=1
DD=11
hh=10
mm=0
ss=0
import time as t
#a, e, i, Omega, omega, L
#AU, , °, °, °, °
bigdi=GP()
EarthE=bigdi["Earth"]
SaturnE=bigdi["Pluto"]
G = 6.67*10**(-11)
muS = G*1.989*10**(30) #km3/s2
AU2km = 149597870.7
km2AU = 1/AU2km
RealmuS = muS*km2AU**3
start=t.time()
n=185
dV=np.zeros((n,))*np.nan
dt=np.linspace(1,750,n)
for i in range(n):
    print(i)
    xE,xS,vE,vS=PP(YY,MM,DD,hh,mm,ss,EarthE,SaturnE,dt[i])
    dtheta=getangle(xE,xS)
    tm=1
    r1=xE
    r2=xS
    O2=Project(dtheta=dtheta, dt=dt[i], nrev=0,LoS=0,tm=tm,mu=1,r1=r1,r2=r2)
    try:
        dV[i]=np.linalg.norm(np.abs(O2.v1-vE*km2AU)+np.abs(vS*km2AU-O2.v2))
    except AttributeError:  
        continue
stop=t.time()
i=np.where(dV==np.nanmin(dV))[0][0]
print('Exec time :',stop-start)
print(np.where(dV==np.nanmin(dV)))
print(np.nanmin(dV))
print("dt : ",dt[np.where(dV==np.nanmin(dV))[0][0]])
xE,xS,vE,vS=PP(YY,MM,DD,hh,mm,ss,EarthE,SaturnE,dt[i])
dtheta=getangle(xE,xS)
tm=1
r1=xE
r2=xS
O2=Project(dtheta=dtheta, dt=dt[i], nrev=0,LoS=0,tm=tm,mu=1,r1=r1,r2=r2)
print(O2.a, O2.e, O2.p)

    

