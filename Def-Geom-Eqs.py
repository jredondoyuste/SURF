from re import L
import numpy as np
import matplotlib.pyplot as plt 
import math
from scipy import special
import time
from tqdm import tqdm
import quadpy
import pickle

##############################################
############## Manual Inputs #################
##############################################

#Time Integration

t0 = 100                           #Time at which the (backwards) evolution starts. 
h = 0.2                            #Stepsize of the time integration
Nsteps = int(np.floor(t0 / h))      #Number of steps in the time integration. 

#Source Gaussian Profile

tS = 50
sigmaS = 12

#Excitation Coefficients

coef22 = 0.1    #Coefficient of the l=2, m=2 mode
coef21 = 0.02    #Coefficient of the l=2, m=1 mode
coef20 = 0.05    #Coefficient of the l=2, m=0 mode

#Number of l modes

ltop = 3           #Number of the higher l mode that is considered physical. This does NOT include the ghost modes. 

Initialization = np.array([t0, h, Nsteps, tS, sigmaS, coef22, coef21, coef20, ltop])
headerIni = "t0, h, Nsteps, tS, sigmaS, coef22, coef21, coef20, ltop\n"
np.savetxt('Data/Ini.dat', Initialization, header = headerIni)

##############################################
############## Initialization ################
##############################################

print()
print('------------------------------Initializing code--------------------------------')
print()
print('Numerical Integration: Stepsize = %.3f' % (h) + ', Physical modes up to l = %.0f' %ltop)
print()
print('Source excitation: Coef (2,2) = %.2f ' % coef22 + ', Coef (2,1) = %.2f ' %coef21 + ', Coef (2,0)= %.2f' %coef20)
print()
print('Source hits at time = %.2f' %tS + ', With Gaussian width = %.2f' % sigmaS)
print()

start_ini = time.time()

# Definition of the spin-weighted spherical harmonics. 

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

# Eth prefactors

def eth(l,s):
    if l < abs(s):
        return 0
    else:
        return np.sqrt( (l - s) * (l + s + 1) )
def ethbar(l,s):
    if l < abs(s):
        return 0
    else:
        return -np.sqrt( (l + s) * (l - s + 1))

# Function to turn from j to (l, m)

def jtolm(j):
    if isinstance(np.sqrt(j), int):
        return np.sqrt(j), j - 2 * np.sqrt(j)
    else:
        return np.floor( np.sqrt(j) ), j - np.floor( np.sqrt(j) ) * ( np.floor( np.sqrt(j) ) + 1 )

def lmtoj(l, m):
    return l ** 2 + l + m 

# Include ghost modes, construct array

#ghost = int(np.floor(ltop /2))
ghost = 1
lmax = ltop + ghost
jmax = lmtoj(lmax, lmax) + 1
jghosts = np.linspace(lmtoj(ltop, ltop) + 1, jmax-1, jmax-lmtoj(ltop, ltop)).astype(int)
    
##############################################
############## Import angular integrals ######
##############################################

I000 = pickle.load( open('I000.pkl', 'rb') )
I1m10 = pickle.load( open('I1m10.pkl', 'rb'))
I2m20 = pickle.load( open('I2m20.pkl', 'rb') )
I101 = pickle.load( open('I101.pkl', 'rb') )
I2m11 = pickle.load( open('I2m11.pkl', 'rb') )
I202 = pickle.load( open('I202.pkl', 'rb') )
I112 = pickle.load( open('I112.pkl', 'rb') )

print('All angular integrals imported' + '----%.2f---seconds' % (time.time() - start_ini))
print()
##############################################
############## Source Characteristics ########
##############################################

freqs = np.zeros(jmax, dtype=np.cdouble)
coefsL = np.zeros(jmax, dtype=np.cdouble)

# QNMs frequencies extracted from https://pages.jh.edu/eberti2/ringdown/ http://arxiv.org/abs/0905.2975  n = 0

freqs[lmtoj(2,2)] = (0.7473433688360838 + 0.1779246313778714 * 1j)
freqs[lmtoj(2,1)] = (0.4965265283562174 + 0.1849754359058844 * 1j)
freqs[lmtoj(2,0)] = (0.2209098781608393 + 0.2097914341737619 * 1j)
freqs[lmtoj(2,-2)] = (0.7473433688360838 + 0.1779246313778714 * 1j)
freqs[lmtoj(2,-1)] = (0.4965265283562174 + 0.1849754359058844 * 1j)
coefsL[lmtoj(2,2)] = coef22
coefsL[lmtoj(2,1)] = coef21
coefsL[lmtoj(2,0)] = coef20
coefsL[lmtoj(2,-2)] = coef22
coefsL[lmtoj(2,-1)] = coef21

t_shifted = tS + (sigmaS ** 2 ) * np.imag(freqs[lmtoj(2,2)])

def source_not_norm(t, j):
    # if (t - tS) ** 2 > (3 * sigmaS) ** 2:
    #     return 0
    # else:
    return (coefsL[j] * np.exp(1j * freqs[j] * t) * np.exp(- ((t - t_shifted) ** 2) / (2 * (sigmaS ** 2))))

t_normalize = np.linspace(0,t0,t0)

norm_cst = 10 ** (-6)
for t in t_normalize:
    for j in range(jmax):
        norm_maybe = np.abs(source_not_norm(t, j))
        if norm_maybe > norm_cst:
            norm_cst = norm_maybe

def source(t, j):
    return source_not_norm(t, j) / (10 * norm_cst)

def der_source(t, j):
    # if (t - tS) ** 2 > (3 * sigmaS) ** 2:
    #     return 0
    # else:
    return ((1j * freqs[j] - ((t - t_shifted) / ((sigmaS ** 2)))) * coefsL[j] * np.exp(1j * freqs[j] * t) * np.exp(- ((t - t_shifted) ** 2) / (2 * (sigmaS ** 2)))) / (10 * norm_cst)

##############################################
############## Evolution Equations ###########
##############################################

# Expansion Equation

def thRHS(t, y, bigJ):       
    linear_term = y[bigJ] / 2                       
    nl1, nl2, nl3, nl4, nl5 = 0, 0, 0, 0, 0
    for j1 in range(jmax):
        l1, m1 = jtolm(j1)
        for j2 in range(jmax):
            l2, m2 = jtolm(j2)
            I00 = I000[j1, j2, bigJ]
            I22 = I2m20[j1, j2, bigJ]
            nl1 += - (3/2) * y[j1] * y[j2] * I00    
            nl2 += - (1/4) * y[j1] * (ethbar(l2, 1) * y[j2 + jmax] + eth(l2, -1) * np.conjugate(y[j2 + jmax])) * I00  
            #nl3 += - eth(l1, 1) * y[j1 + jmax] * ethbar(l2, -1) * np.conjugate(y[j2 + jmax]) * I22
            nl4 += - der_source(t, j1) * np.conjugate(der_source(t, j2)) * I22
            nl5 += - (1/4) * source(t, j1) * np.conjugate(source(t, j2)) * I22
    return np.real(linear_term + nl1 + nl2 + nl3 + nl4 + nl5)

# Hajiceck equation

def HRHS(t, y, bigJ):
    bigL, bigM = jtolm(bigJ)
    #linear_1 = (1/4) * (eth(bigL, 0) * ethbar(bigL, 1)  + ethbar(bigL, 2) * eth(bigL, 1)) * y[bigJ + jmax]
    linear_2 = - (1/2) * eth(bigL, 0) * y[bigJ]
    linear_3 = - (1/4) * ethbar(bigL, 2) * source(t, bigJ)
    linear_4 = - (1/2) * ethbar(bigL, 2) * der_source(t, bigJ)
    linear_1 = 0
    nl1, nl2, nl3, nl4 = 0, 0, 0, 0
    for j1 in range(jmax):
        l1, m1 = jtolm(j1)
        for j2 in range(jmax):
            l2, m2 = jtolm(j2)
            I10 = I101[j1, j2, bigJ]
            I21 = I2m11[j1, j2, bigJ]
            nl1 += y[j1 + jmax] * ethbar(l2, 1) * y[j2 + jmax] * I10
            nl2 += - y[j1 + jmax] * y[j2] * I10
            nl3 += (1/4) * ethbar(l1, 2) * source(t, j1) * y[j2] * I10
            nl4 += (1/2) * source(t, j1) * ethbar(l2, 0) * y[j2] * I21
    return linear_1 + linear_2 + linear_3 + linear_4 + nl1 + nl2 + nl3 + nl4

##############################################
############## Surface Gravity ###############
##############################################

def Onefun(j):
    def foo(theta_phi):
        l, m = jtolm(j)
        theta, phi = theta_phi
        return np.conjugate(yslm(0,l, m, theta, phi))
    return foo

One = []
for j in range(jmax):
    One.append(scheme.integrate_spherical(Onefun(j)))

def sur_grav(t, y):
    kap = []
    for bigJ in range(jmax):
        bigL, bigM = jtolm(bigJ)
        term1 = (1/2) * One[bigJ] 
        term2 = - y[bigJ]
        term3 = - (1/2) * eth(bigL, -1) * np.real(y[bigJ + jmax]) 
        nl = 0
        for j1 in range(jmax):
            for j2 in range(jmax):
                nl += -(1/2) * y[j1 + lmax] * np.conjugate(y[j2 + jmax]) * I1m10[j1, j2, bigJ]
        kap.append(np.real(term1 + term2 + term3 + nl))
    return kap

##############################################
################# Shear ######################
##############################################

def shear(t, y, kp):
    sh = []
    for bigJ in range(jmax):
        bigL, bigM = jtolm(bigJ)
        term1 = der_source(t, bigJ)
        term2 = - (1/2) * eth(bigL, 1) * y[bigJ + jmax]
        nl1 = 0
        nl2 = 0
        nl3 = 0
        for j1 in range(jmax):
            for j2 in range(jmax):
                I20 = I202[j1, j2, bigJ]
                I11 = I112[j1, j2, bigJ]
                nl1 += kp[j2] * source(t, j1) * I20
                nl2 += - (1/2) * y[j2] * source(t, j1) * I20
                nl3 += - y[j1 + jmax] * y[j2 + jmax] * I11
        sh.append(term1 + term2 + nl1 + nl2 + nl3)
    return sh

##############################################
################# Energy density #############
##############################################

def energy(t, y, kp, bigJ):
    lin1 = y[bigJ]
    nl = 0
    for j1 in range(jmax):
        ktilde = kp[j1] - One[j1] / 2
        for j2 in range(jmax):
            nl += - y[j2] * ktilde * I000[j1, j2, bigJ]
    return np.real(lin1 + nl)

##############################################
################# Pressure ###################
##############################################

def pressure(t, y, kp, bigJ):
    bigL, bigM = jtolm(bigJ)
    linear1 = (1/2) * (kp[bigJ] - One[bigJ] / 2)
    linear2 = - (1/2) * y[bigJ]
    linear3 = - thRHS(t, y, bigJ) - ethbar(bigL, 1) * np.real(HRHS(t, y, bigJ)) 
    nl1, nl2, nl3, nl4 = 0, 0, 0, 0
    for j1 in range(jmax):
        ktilde = kp[j1] - One[j1] / 2
        for j2 in range(jmax):
            l2, m2 = jtolm(j2)
            nl1 += -(1/2) * np.real((2 * eth(l2, 0) * y[j2] + ethbar(l2, 2) * source(t, j2) + ethbar(l2, 2) * der_source(t, j2)) * y[j1 + jmax] * I1m10[j1, j2, bigJ])
            nl2 += 3 * ktilde * (y[j2] / 2 - (1/4) * ethbar(l2, 1) * np.real(2 * eth(l2, 0) * y[j2] + ethbar(l2, 2) * source(t, j2) + ethbar(l2, 2) * der_source(t, j2)) ) * I000[j1, j2, bigJ]
            nl3 += - ktilde * (kp[j2] - One[j2] / 2) * I000[j1, j2, bigJ]
            nl4 += (1/2) * ktilde * y[j2] * I000[j1, j2, bigJ]
    return np.real(One[bigJ] / 2 + linear1 + linear2 + linear3 + nl1 + nl2 + nl3 + nl4)

##############################################
################# Dissipation ################
##############################################

def dissipation(t, kp, sh, bigJ):
    lin1 = - sh[bigJ]
    nl = 0
    for j1 in range(jmax):
        ktilde = kp[j1] - One[j1] / 2
        for j2 in range(jmax):
            nl += sh[j2] * ktilde * I202[j1, j2, bigJ]
    return lin1 + nl

##############################################
################# Heat Flux  #################
##############################################

def heat(t, y, kp, sh, bigJ):
    bigL, bigM = jtolm(bigJ)
    linear = 2 * eth(bigL, 0) * kp[bigJ]
    nl1, nl2, nl3 = 0, 0, 0
    for j1 in range(jmax):
        ktilde = kp[j1] - One[j1] / 2
        for j2 in range(jmax):
            l2, m2 = jtolm(j2)
            nl1 += - ktilde * eth(l2, 0) * kp[j2] * I101[j2, j1, bigJ]
            nl2 += (1/2) * sh[j1] * np.conjugate(y[j2 + jmax]) * I2m11[j1, j2, bigJ]
            nl3 += (1/4) * y[j1 + jmax] * ethbar(l2, 1) * np.real(2 * eth(l2, 0) * y[j2] + ethbar(l2, 2) * source(t, j2) + ethbar(l2, 2) * der_source(t, j2)) * I101[j1, j2, bigJ]
    return - (1/2) * (linear + nl1 + nl2 + nl3)

##############################################
################# Vorticity  #################
##############################################

def vorticity(t, y, kp, bigJ):
    bigL, bigM = jtolm(bigJ)
    linear =  eth(bigL, 0) * np.conjugate(y[bigJ + jmax])
    nl1, nl2, nl3 = 0, 0, 0
    for j1 in range(jmax):
        ktilde = kp[j1] - One[j1] / 2
        for j2 in range(jmax):
            l2, m2 = jtolm(j2)
            nl1 += - ktilde * eth(l2, 0) * np.conjugate(y[j2 + jmax]) * I000[j1, j2, bigJ]
            nl2 += 2 * y[j1 + jmax] * ethbar(l2, 0) * kp[j2] * I1m10[j1, j2, bigJ]
            nl3 += - (1/4) * y[j1 + jmax] * np.conjugate(2 * eth(l2, 0) * y[j2] + ethbar(l2, 2) * source(t, j2) + ethbar(l2, 2) * der_source(t, j2)) * I1m10[j1, j2, bigJ]
    return  (1/2) * (linear + nl1 + nl2 + nl3)

##############################################
################# Enstrophy  #################
##############################################

def enstrophy(vor, bigJ):
    nl = 0
    for j1 in range(jmax):
        for j2 in range(jmax):
            nl += vor[j1] * vor[j2] * I000[j1, j2, bigJ]
    return  nl

##############################################
############## RK Algorithm ##################
##############################################

def RKstep(t, y, h):
    k1 = np.zeros(2 * jmax, dtype = np.cdouble)
    k2 = np.zeros(2 * jmax, dtype = np.cdouble)
    k3 = np.zeros(2 * jmax, dtype = np.cdouble)
    k4 = np.zeros(2 * jmax, dtype = np.cdouble)
    for j in range(jmax):
        k1[j] = thRHS(t, y, j)
        k1[j + jmax] = HRHS(t, y, j)
    y2 = y - k1 * h / 2
    for j in range(jmax):
        k2[j] = thRHS(t - h/2, y2, j)
        k2[j + jmax] = HRHS(t - h/2, y2, j)
    y3 = y - k2 * h / 2
    for j in range(jmax):
        k3[j] = thRHS(t - h/2, y3, j)
        k3[j + jmax] = HRHS(t - h/2, y3, j)
    y4 = y - k3 * h
    for j in range(jmax):
        k4[j] = thRHS(t - h, y4, j)
        k4[j + jmax] = HRHS(t - h, y4, j)
    return y - h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

##############################################
############## Evolution! ####################
##############################################

# Initial Condition

y0 = np.zeros(2 * jmax, dtype = np.cdouble)
yini = y0

th_evol = np.zeros((Nsteps, jmax), dtype = np.cdouble)
H_evol = np.zeros((Nsteps, jmax), dtype = np.cdouble)
k_evol = np.zeros((Nsteps, jmax), dtype = np.cdouble)
sh_evol = np.zeros((Nsteps, jmax), dtype = np.cdouble)
en_evol = np.zeros((Nsteps, jmax), dtype = np.cdouble)
pr_evol = np.zeros((Nsteps, jmax), dtype = np.cdouble)
dis_evol = np.zeros((Nsteps, jmax), dtype = np.cdouble)
pi_evol = np.zeros((Nsteps, jmax), dtype = np.cdouble)
vor_evol = np.zeros((Nsteps, jmax), dtype = np.cdouble)
ens_evol = np.zeros((Nsteps, jmax), dtype = np.cdouble)
t_evol = [] 
s_evol = np.zeros((Nsteps, jmax), dtype = np.cdouble)
ds_evol = np.zeros((Nsteps, jmax), dtype = np.cdouble)

for step in tqdm(range(Nsteps)):
    t = t0 - step * h
    yinter = RKstep(t, yini, h)
    yfinal = yinter
    scale = np.amax(np.abs(yfinal))
    for j in jghosts:
        if np.abs(yfinal[j]) > 0.1 * scale:
            print("Ghost modes too large")
            break
        yfinal[j] = 0
        yfinal[j + jmax] = 0
    yini = yfinal
    kp = sur_grav(t, yfinal)
    sh = shear(t, yfinal, kp)
    k_evol[step, :] = kp
    sh_evol[step, :] = sh
    th_evol[step, :] = yfinal[0:jmax]
    H_evol[step, :] = yfinal[jmax: 2 * jmax]
    t_evol.append(t)
    for j in range(jmax):
        s_evol[step, j] = source(t, j)
        ds_evol[step, j] = der_source(t, j)
        en_evol[step, j] = energy(t, yfinal, kp, j)
        pr_evol[step, j] = pressure(t, yfinal, kp, j)
        dis_evol[step, j] = dissipation(t, kp, sh, j)
        pi_evol[step, j] = heat(t, yfinal, kp, sh, j)
        vor_evol[step, j] = vorticity(t, yfinal, kp, j)
    for j in range(jmax):
        ens_evol[step, j] = enstrophy(vor_evol[step, :], j)
    if np.sum(np.abs(yfinal)) > 10 ** (6):
        print('All too big')
        break

##############################################
############## Save the data #################
##############################################

headerK = "Evolution of the extrinsic curvature \n"
np.savetxt('Data/Kappa.dat', k_evol, header = headerK)
headerTh = "Evolution of the expansion \n"
np.savetxt('Data/Expansion.dat', th_evol, header = headerTh)
headerN = "Evolution of the shear \n"
np.savetxt('Data/Shear.dat', sh_evol, header = headerK)
headerH = "Evolution of the Hajiceck \n"
np.savetxt('Data/Hajiceck.dat', H_evol, header = headerH)
headerEn = "Fluid Energy \n"
np.savetxt('Data/Energy.dat', en_evol, header = headerEn)
headerPr = "Fluid Pressure \n"
np.savetxt('Data/Pressure.dat', pr_evol, header = headerPr)
headerDis = "Fluid Dissipation \n"
np.savetxt('Data/Dissipation.dat', dis_evol, header = headerDis)
headerPi = "Fluid Heat \n"
np.savetxt('Data/Heat.dat', pi_evol, header = headerPi)
headerV = "Fluid Vorticity \n"
np.savetxt('Data/Vorticity.dat', vor_evol, header = headerV)
headerEns = "Fluid Enstrophy \n"
np.savetxt('Data/Enstrophy.dat', ens_evol, header = headerEns)
headerS = "Source\n"
np.savetxt('Data/Source.dat', s_evol, header = headerS)
headerdS = "Derivative Source\n"
np.savetxt('Data/DerSource.dat', ds_evol, header = headerS)
headerT = "Time vector\n"
np.savetxt('Data/Time.dat', t_evol, header = headerT)