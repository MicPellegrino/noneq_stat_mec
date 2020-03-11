import math
import numpy as np
import numpy.random
import matplotlib.pyplot as plt
from numpy import linalg

# Basis
e = np.array([1.0, 0.0])
g = np.array([0.0, 1.0])

# Initial state
psi_0 = ( e + g ) / np.sqrt(2.0)
# psi_0 = e

# Operators
sigma_z = np.array([[1.0, 0.0],[0.0, -1.0]])
sigma_p = np.array([[0.0, 1.0],[0.0, 0.0]])
sigma_m = np.array([[0.0, 0.0],[1.0, 0.0]])
sigma_pm = np.matmul(sigma_p, sigma_m)
sigma_mp = np.matmul(sigma_m, sigma_p)

# Parameters
omega = 3.0
gamma = 1.0
n = 10

# Time-step
dt = 0.002
t_fin = 0.5
K = int(t_fin/dt)+1
t = np.linspace(0,t_fin,K)

# Continous evolution
J = 0.5*omega*sigma_z - 0.5*1j*gamma*( (n+1)*sigma_pm + n*sigma_mp )
I = np.eye(2, dtype=complex)
L_cont = I - 1j*J*dt

# Jumps
L_p = np.sqrt(gamma*n)*sigma_p
L_m = np.sqrt(gamma*(n+1))*sigma_m
L_pdp = gamma*n*sigma_mp
L_mdm = gamma*(n+1)*sigma_pm
L_sum = L_pdp + L_mdm

# Utilities
inner3 = lambda x, A, y : np.vdot(x, np.dot(A,y))
prob_p = lambda psi : (inner3(psi,L_pdp,psi)*dt).real
prob_m = lambda psi : (inner3(psi,L_mdm,psi)*dt).real
prob_0 = lambda psi : 1 - prob_p(psi) - prob_m(psi)

def cumul_prob (phi):
    cp = np.zeros(3)
    cp[1] = prob_p(phi)
    cp[2] = prob_m(phi)
    cp[0] = 1 - np.sum(cp)
    cp = np.cumsum(cp)
    return cp

def mc_outcome (phi):
    r = np.random.uniform()
    cp = cumul_prob(phi)
    for i in range(3):
        if r < cp[i]:
            return i

# Number of samples
M = 5000
psi_ens_new = [psi_0]*M

# Data structures
re_rho = [[],[]]
re_rho[0].append( np.zeros(K, dtype=float) )
re_rho[0].append( np.zeros(K, dtype=float) )
re_rho[1].append( np.zeros(K, dtype=float) )
re_rho[1].append( np.zeros(K, dtype=float) )
im_rho = [[],[]]
im_rho[0].append( np.zeros(K, dtype=float) )
im_rho[0].append( np.zeros(K, dtype=float) )
im_rho[1].append( np.zeros(K, dtype=float) )
im_rho[1].append( np.zeros(K, dtype=float) )

def average_density(state_ens) :
    M = len(state_ens)
    rho = np.zeros(2, dtype=complex)
    for i in range(M):
        rho = rho + np.outer(state_ens[i].conjugate(),state_ens[i])
    return rho/M

rho = average_density(psi_ens_new)
re_rho[0][0][0] = rho[0,0].real
re_rho[0][1][0] = rho[0,1].real
re_rho[1][0][0] = rho[1,0].real
re_rho[1][1][0] = rho[1,1].real
im_rho[0][0][0] = rho[0,0].imag
im_rho[0][1][0] = rho[0,1].imag
im_rho[1][0][0] = rho[1,0].imag
im_rho[1][1][0] = rho[1,1].imag

psi_ens_old = [psi_0]*M

# Loop
for k in range(1,K):
    if k%10 == 0:
        print('step k = '+str(k)+'/'+str(K))
    for i in range(M) :
        s = mc_outcome(psi_ens_old[i])
        if s == 0 :
            psi_new = np.dot(L_cont,psi_ens_old[i])
        elif s == 1:
            psi_new = np.dot(L_p,psi_ens_old[i])
        else :
            psi_new = np.dot(L_m,psi_ens_old[i])
        # psi_new = psi_new/np.vdot(psi_new, psi_new)
        psi_new = psi_new/np.linalg.norm(psi_new)
        psi_ens_new[i] = psi_new
    rho = average_density(psi_ens_new)
    re_rho[0][0][k] = rho[0,0].real
    re_rho[0][1][k] = rho[0,1].real
    re_rho[1][0][k] = rho[1,0].real
    re_rho[1][1][k] = rho[1,1].real
    im_rho[0][0][k] = rho[0,0].imag
    im_rho[0][1][k] = rho[0,1].imag
    im_rho[1][0][k] = rho[1,0].imag
    im_rho[1][1][k] = rho[1,1].imag
    psi_ens_old = psi_ens_new

# Analytical expression
# rho_ee_analytic = np.exp(-gamma*t)

# Save to file
with open('Quan/re_rho_00.txt', 'w') as f:
    for item in re_rho[0][0]:
        f.write("%s\n" % item)
with open('Quan/re_rho_01.txt', 'w') as f:
    for item in re_rho[0][1]:
        f.write("%s\n" % item)
with open('Quan/re_rho_10.txt', 'w') as f:
    for item in re_rho[1][0]:
        f.write("%s\n" % item)
with open('Quan/re_rho_11.txt', 'w') as f:
    for item in re_rho[1][1]:
        f.write("%s\n" % item)
with open('Quan/im_rho_01.txt', 'w') as f:
    for item in im_rho[0][1]:
        f.write("%s\n" % item)
with open('Quan/im_rho_10.txt', 'w') as f:
    for item in im_rho[1][0]:
        f.write("%s\n" % item)

# Plotting
# plt.plot(t, re_rho_ee, 'r-', label=r'Re[$\rho_{ee}$]')
# plt.plot(t, im_rho_ee, 'b-', label=r'Im[$\rho_{ee}$]')
# plt.plot(t, rho_ee_analytic, 'k--')
plt.plot(t, re_rho[0][0], label=r'$\rho_{ee}$')
plt.plot(t, re_rho[1][1], label=r'$\rho_{gg}$')
plt.plot(t, re_rho[0][1], label=r'Re[$\rho_{eg}$]')
plt.plot(t, re_rho[1][0], label=r'Re[$\rho_{ge}$]')
plt.plot(t, im_rho[0][1], label=r'Im[$\rho_{eg}$]')
plt.plot(t, im_rho[1][0], label=r'Im[$\rho_{ge}$]')
plt.plot(t, im_rho[0][0]+im_rho[1][1], 'k--')
plt.plot(t, re_rho[0][0]+re_rho[1][1], 'r--')
plt.legend(fontsize=20.0)
plt.xlim([0,t_fin])
plt.xlabel(r'$t$', fontsize=20.0)
plt.ylabel(r'$\rho(t)$', fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.title(r'$\omega$='+str(omega)+r', $\Gamma$='+str(gamma)+r', $n$='+str(n), fontsize=20.0)
plt.show()
