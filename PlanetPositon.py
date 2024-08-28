# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 11:29:46 2023

@author: lhoes
"""

import numpy as np
deg2rad=np.pi/180
precision = 10**-16

def GP():
    GP={
        "Earth" : np.array([[1.00000261, 0.01671123, -0.00001531, 0, 102.93768193, 100.46457166],
                         [0.00000562, -0.00004392, -0.01294668, 0, 0.32327364, 35999.37244981]]),
        "Saturn" : np.array([[9.53667594, 0.05386179, 2.48599187, 113.66242448, 92.59887831, 49.95424423],
                          [-0.0012506, -0.00050991, 0.00193609, -0.28867794, -0.41897216, 1222.49362201]]),
        "Mercury" : np.array([[0.38709927, 0.20563593, 7.00497902, 48.33076593, 77.45779628, 252.25032350],
                          [0.00000037, 0.00001906, -0.00594749, -0.12534081, 0.16047689, 149472.67411175]]),
        "Venus" : np.array([[0.72333566, 0.00677672, 3.39467605, 76.67984255, 131.60246718, 181.97909950],
                          [0.00000390, -0.00004107, -0.00078890, -0.27769418, 0.00268329, 58517.81538729]]),
        "Mars" : np.array([[1.52371034, 0.09339410, 1.84969142, 49.55953891, -23.94362959, -4.55343205],
                          [0.00000390, -0.00004107, -0.00078890, -0.27769418, 0.00268329, 58517.81538729]]),
        "Jupiter" : np.array([[5.20288700, 0.04838624, 1.30439695, 100.47390909, 14.72847983, 34.39644501],
                          [-0.00011607, -0.00013253, -0.00183714, 0.20469106, 0.21252668, 3034.74612775]]),
        "Uranus" : np.array([[19.18916464, 0.04725744, 0.77263783, 74.01692503, 170.95427630, 313.23810451],
                          [-0.00196176, -0.00004397, -0.77263783, 0.04240589, 0.40805281, 428.48202785]]),
        "Neptune" : np.array([[30.06992276, 0.00859048, 1.77004347, 131.78422574, 44.96476227, -55.12002969],
                          [0.00026291, 0.00005105, 0.00035372, -0.00508664, -0.32241464, 218.45945325]]),
        "Pluto" : np.array([[39.48211675, 0.24882740, 17.14001206, 110.30393684, 224.06891629, 238.92903833],
                          [-0.00031596, 0.00005170, 0.00004818, -0.01183482, -0.04062942, 145.20780515]])
        
        }
    return GP

def PP(YY,MM,DD,hh,mm,ss,EarthE,SaturnE,dt):
    jd,fr=JD(YY,MM,DD,hh,mm,ss)
    jc=JC(jd+fr)
    jd2,fr2=JD(YY,MM,DD+dt,hh,mm,ss)
    jc2=JC(jd2+fr2)
    pEE = Propagate(jc, EarthE)
    pSE = Propagate(jc2, SaturnE)
    G = 6.67*10**(-11)
    muS = G*1.989*10**(30) #km3/s2

    hE = (muS*pEE[0]*(1-pEE[1]**2))**(1/2)
    hS = (muS*pSE[0]*(1-pSE[1]**2))**(1/2)

    wE = pEE[4]-pEE[3]
    ME = pEE[5] - pEE[4]
    wS = pSE[4]-pSE[3]
    MS = pSE[5] - pSE[4]

    Eo=0
    e=pEE[1]
    M=ME
    EE = kepler(Eo,e,M)

    e=pSE[1]
    M=MS
    SE = kepler(Eo,e,M)
    
    E0 = teta(EE,pEE[1])
    S0 = teta(SE,pSE[1])

    muE = G * 5.972*10**(24)
    muSat = G * 5.683*10**26
    
    #a, e, i, Omega, omega, L
    xE, vE = oa2rv(pEE[0],pEE[1],pEE[3],pEE[2],wE*deg2rad,E0,muS,hE)
    xS, vS = oa2rv(pSE[0],pSE[1],pSE[3],pSE[2],wS*deg2rad,S0,muS,hS)
    
    return xE,xS,vE,vS
    

def JD(YY,MM,DD,hh=0,mm=0,ss=0):
    if MM<=2 : 
        YY=YY-1
        MM=MM+12
    day=int(365.25*YY)+int(30.6001*(MM+1))+DD+1720981.5
    fr=(hh+mm/60+ss/3600)/24
    return (day,fr)



def JC(jd):
    return (jd-2451545)/36525



def Propagate(jc,obj):
    (shpx,shpy)=obj.shape
    prop = np.zeros((shpy,))
    
    for i in range(shpy):
        prop[i]=obj[0,i]+obj[1,i]*jc

    return prop



def f(E,e,M):
    return E-e*np.sin(E*deg2rad)-M

def fp(E,e):
    return 1-e*np.cos(E*deg2rad)

def kepler(Eo,e,M):
    n=0
    while 1: 
        E=Eo-f(Eo,e,M)/fp(Eo,e)
        n+=1
        if np.abs(E-Eo)<precision or n>20:
            break
        Eo=E
    return E



def teta(E,e):
    return np.arctan(((1+e)/(1-e))**(1/2)*np.tan((E*deg2rad)/2))*2


def R1(o):
    return np.array([[1,         0,          0],
                     [0, np.cos(o), -np.sin(o)],
                     [0, np.sin(o),  np.cos(o)]])

def R3(o):
    return np.array([[np.cos(o), -np.sin(o), 0],
                     [np.sin(o), np.cos(o),  0],
                     [        0,         0,  1]])


def oa2rv (a,e,Omega,i,w,theta,mu,h):
    # p=a*(1-e**2)
    # h=p/mu #C'est h²
    r= (h**2)/mu*1/(1+e*np.cos(theta))
    r=np.array([[r*np.cos(theta)],
                [r*np.sin(theta)],
                [0]])
    v=mu/h*np.array([[-np.sin(theta)],
                     [e+np.cos(theta)],
                     [0]])
    R2 = R3(-Omega*deg2rad)@R1(-i*deg2rad)@R3(-w*deg2rad)
    
    x = R2@r
    v = R2@v
    return (x.reshape((3,)),v.reshape((3,)))

