import numpy as np
import math
from scipy import special
import quadpy
from tqdm import tqdm
import pickle

# Code to compute and save all of the angular integrals, up to a high enough value of J = (L, M). 

lmax = 12

def jtolm(j):
    if isinstance(np.sqrt(j), int):
        return np.sqrt(j), j - 2 * np.sqrt(j)
    else:
        return np.floor( np.sqrt(j) ), j - np.floor( np.sqrt(j) ) * ( np.floor( np.sqrt(j) ) + 1 )

def lmtoj(l, m):
    return l ** 2 + l + m

jmax = lmtoj(lmax, lmax) + 1

def yslm(s,l,m,theta,phi):
    if l<abs(s):
        return 0
    elif s>=0:
        prefactor = (-1)**m * (math.factorial(int(l+m))*math.factorial(int(l-m))*(2*l+1)/(4*np.pi*math.factorial(int(l+s))*math.factorial(int(l-s))))**(1/2)
        prefactor2 = (np.sin(theta/2))**(2*l)
        factor = 0
        for kk in range(int(l-s+1)):
            factor +=  special.binom(l-s,kk)*special.binom(l+s,kk+s-m)*(-1)**(l-s-kk)*np.exp(1j*m*phi)*(np.cos(theta/2)/np.sin(theta/2))**(2*kk+s-m)
        return prefactor*prefactor2*factor
    else:
        newfactor = (-1)**(s+m)
        m = -m
        s = -s
        prefactor = (-1)**m * (math.factorial(int(l+m))*math.factorial(int(l-m))*(2*l+1)/(4*np.pi*math.factorial(int(l+s))*math.factorial(int(l-s))))**(1/2)
        prefactor2 = (np.sin(theta/2))**(2*l)
        factor = 0
        for kk in range(int(l-s+1)):
            factor +=  special.binom(l-s,kk)*special.binom(l+s,kk+s-m)*(-1)**(l-s-kk)*np.exp(1j*m*phi)*(np.cos(theta/2)/np.sin(theta/2))**(2*kk+s-m)
        return newfactor*prefactor*prefactor2*factor
    
scheme = quadpy.u3.get_good_scheme(8)

def Y2Ym2Y0(l1, m1, l2, m2, bigL, bigM):
    def foo(theta_phi):
        theta, phi = theta_phi
        return yslm(2,l1, m1,theta, phi)*yslm(-2,l2, m2,theta, phi)*np.conjugate(yslm(0,bigL, bigM,theta, phi))
    return foo

I2m20 = np.zeros((jmax, jmax, jmax), dtype = np.cdouble)
for j1 in tqdm(range(jmax)):
    l1, m1 = jtolm(j1)
    for j2 in range(jmax):
        l2, m2 = jtolm(j2)
        for bigJ in range(jmax):
            bigL, bigM = jtolm(bigJ)
            I2m20[j1, j2, bigJ] = scheme.integrate_spherical(Y2Ym2Y0(l1, m1, l2, m2, bigL, bigM))

def Y1Ym1Y0(l1, m1, l2, m2, bigL, bigM):
    def foo(theta_phi):
        theta, phi = theta_phi
        return yslm(1,l1, m1,theta, phi)*yslm(-1,l2, m2,theta, phi)*np.conjugate(yslm(0,bigL, bigM,theta, phi))
    return foo

I1m10 = np.zeros((jmax, jmax, jmax), dtype = np.cdouble)
for j1 in tqdm(range(jmax)):
    l1, m1 = jtolm(j1)
    for j2 in range(jmax):
        l2, m2 = jtolm(j2)
        for bigJ in range(jmax):
            bigL, bigM = jtolm(bigJ)
            I1m10[j1, j2, bigJ] = scheme.integrate_spherical(Y1Ym1Y0(l1, m1, l2, m2, bigL, bigM))

def Y0Y0Y0(l1, m1, l2, m2, bigL, bigM):
    def foo(theta_phi):
        theta, phi = theta_phi
        return yslm(0,l1, m1,theta, phi)*yslm(0,l2, m2,theta, phi)*np.conjugate(yslm(0,bigL, bigM,theta, phi))
    return foo

I000 = np.zeros((jmax, jmax, jmax), dtype = np.cdouble)
for j1 in tqdm(range(jmax)):
    l1, m1 = jtolm(j1)
    for j2 in range(jmax):
        l2, m2 = jtolm(j2)
        for bigJ in range(jmax):
            bigL, bigM = jtolm(bigJ)
            I000[j1, j2, bigJ] = scheme.integrate_spherical(Y0Y0Y0(l1, m1, l2, m2, bigL, bigM))

def Y1Y0Y1(l1, m1, l2, m2, bigL, bigM):
    def foo(theta_phi):
        theta, phi = theta_phi
        return yslm(1,l1, m1,theta, phi)*yslm(0,l2, m2,theta, phi)*np.conjugate(yslm(1,bigL, bigM,theta, phi))
    return foo

I101 = np.zeros((jmax, jmax, jmax), dtype = np.cdouble)
for j1 in tqdm(range(jmax)):
    l1, m1 = jtolm(j1)
    for j2 in range(jmax):
        l2, m2 = jtolm(j2)
        for bigJ in range(jmax):
            bigL, bigM = jtolm(bigJ)
            I101[j1, j2, bigJ] = scheme.integrate_spherical(Y1Y0Y1(l1, m1, l2, m2, bigL, bigM))

def Y2Ym1Y1(l1, m1, l2, m2, bigL, bigM):
    def foo(theta_phi):
        theta, phi = theta_phi
        return yslm(2,l1, m1,theta, phi)*yslm(-1,l2, m2,theta, phi)*np.conjugate(yslm(1,bigL, bigM,theta, phi))
    return foo

I2m11 = np.zeros((jmax, jmax, jmax), dtype = np.cdouble)
for j1 in tqdm(range(jmax)):
    l1, m1 = jtolm(j1)
    for j2 in range(jmax):
        l2, m2 = jtolm(j2)
        for bigJ in range(jmax):
            bigL, bigM = jtolm(bigJ)
            integral = scheme.integrate_spherical(Y2Ym1Y1(l1, m1, l2, m2, bigL, bigM))
            if np.size(integral) > 1:
                I2m11[j1, j2, bigJ] = integral[0]
            else:
                I2m11[j1, j2, bigJ] = scheme.integrate_spherical(Y2Ym1Y1(l1, m1, l2, m2, bigL, bigM))

def Y2Y0Y2(l1, m1, l2, m2, bigL, bigM):
    def foo(theta_phi):
        theta, phi = theta_phi
        return yslm(2,l1, m1,theta, phi)*yslm(0,l2, m2,theta, phi)*np.conjugate(yslm(2,bigL, bigM,theta, phi))
    return foo

I202 = np.zeros((jmax, jmax, jmax), dtype = np.cdouble)
for j1 in tqdm(range(jmax)):
    l1, m1 = jtolm(j1)
    for j2 in range(jmax):
        l2, m2 = jtolm(j2)
        for bigJ in range(jmax):
            bigL, bigM = jtolm(bigJ)
            integral = scheme.integrate_spherical(Y2Y0Y2(l1, m1, l2, m2, bigL, bigM))
            if np.size(integral) > 1:
                I202[j1, j2, bigJ] = integral[0]
            else:
                I202[j1, j2, bigJ] = scheme.integrate_spherical(Y2Y0Y2(l1, m1, l2, m2, bigL, bigM))         

def Y1Y1Y2(l1, m1, l2, m2, bigL, bigM):
    def foo(theta_phi):
        theta, phi = theta_phi
        return yslm(1,l1, m1,theta, phi)*yslm(1,l2, m2,theta, phi)*np.conjugate(yslm(2,bigL, bigM,theta, phi))
    return foo

I112 = np.zeros((jmax, jmax, jmax), dtype = np.cdouble)
for j1 in tqdm(range(jmax)):
    l1, m1 = jtolm(j1)
    for j2 in range(jmax):
        l2, m2 = jtolm(j2)
        for bigJ in range(jmax):
            bigL, bigM = jtolm(bigJ)
            integral = scheme.integrate_spherical(Y1Y1Y2(l1, m1, l2, m2, bigL, bigM))
            if np.size(integral) > 1:
                I112[j1, j2, bigJ] = integral[0]
            else:
                I112[j1, j2, bigJ] = scheme.integrate_spherical(Y1Y1Y2(l1, m1, l2, m2, bigL, bigM)) 


pickle.dump(I000, open('I000.pkl', 'wb'))
pickle.dump(I1m10, open('I1m10.pkl', 'wb'))
pickle.dump(I2m20, open('I2m20.pkl', 'wb'))
pickle.dump(I101, open('I101.pkl', 'wb'))
pickle.dump(I2m11, open('I2m11.pkl', 'wb'))
pickle.dump(I202, open('I202.pkl', 'wb'))
pickle.dump(I112, open('I112.pkl', 'wb'))