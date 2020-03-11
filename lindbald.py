import math
import numpy as np
import matplotlib.pyplot as plt

# Basis
e = np.array([1.0, 0.0])
g = np.array([0.0, 1.0])

# Initial state
phi_0 = ( e + g ) / np.sqrt(2.0)
rho_0 = 0.5 * np.outer(e+g, e+g)

# Operators
sigma_z = np.array([[1.0, 0.0],[0.0, -1.0]])
sigma_p = np.array([[0.0, 1.0],[0.0, 0.0]])
sigma_m = np.array([[0.0, 0.0],[1.0, 0.0]])
sigma_pm = np.matmul(sigma_p, sigma_m)
sigma_mp = np.matmul(sigma_m, sigma_p)

# Parameters
omega = 3.0
gamma = 1.0
N = 10

# Utilities
f_mm = lambda A, B : np.matmul(A, B)
f_comm = lambda A, B : f_mm(A,B)-f_mm(B,A)
f_mm3 = lambda A, B, C : f_mm(f_mm(A,B),C)

# Unitary dynamic and Lindblad operators
L_close = lambda rho : f_comm(sigma_z, rho)
L_emiss = lambda rho : f_mm3(sigma_m,rho,sigma_p) - \
    0.5*( f_mm(sigma_pm,rho) + f_mm(rho,sigma_pm) )
L_absor = lambda rho : f_mm3(sigma_p,rho,sigma_m) - \
    0.5*( f_mm(sigma_mp,rho) + f_mm(rho,sigma_mp) )

# Overall operator acting on rho
L_tot = lambda rho : -0.5*omega*1j*L_close(rho) + \
    gamma*(N+1)*L_emiss(rho) + gamma*N*L_absor(rho)

# Time-step
dt = 0.002
t_fin = 0.5
K = int(t_fin/dt)+1
t = np.linspace(0,t_fin,K)

# Components of rho
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

# Loop
rho_old = rho_0
re_rho[0][0][0] = rho_0[0,0].real
re_rho[0][1][0] = rho_0[0,1].real
re_rho[1][0][0] = rho_0[1,0].real
re_rho[1][1][0] = rho_0[1,1].real
im_rho[0][0][0] = rho_0[0,0].imag
im_rho[0][1][0] = rho_0[0,1].imag
im_rho[1][0][0] = rho_0[1,0].imag
im_rho[1][1][0] = rho_0[1,1].imag
for k in range(1,K):
    rho_new = rho_old + dt*L_tot(rho_old)
    re_rho[0][0][k] = rho_new[0,0].real
    re_rho[0][1][k] = rho_new[0,1].real
    re_rho[1][0][k] = rho_new[1,0].real
    re_rho[1][1][k] = rho_new[1,1].real
    im_rho[0][0][k] = rho_new[0,0].imag
    im_rho[0][1][k] = rho_new[0,1].imag
    im_rho[1][0][k] = rho_new[1,0].imag
    im_rho[1][1][k] = rho_new[1,1].imag
    rho_old = rho_new

# Save to file
with open('Lind/re_rho_00.txt', 'w') as f:
    for item in re_rho[0][0]:
        f.write("%s\n" % item)
with open('Lind/re_rho_01.txt', 'w') as f:
    for item in re_rho[0][1]:
        f.write("%s\n" % item)
with open('Lind/re_rho_10.txt', 'w') as f:
    for item in re_rho[1][0]:
        f.write("%s\n" % item)
with open('Lind/re_rho_11.txt', 'w') as f:
    for item in re_rho[1][1]:
        f.write("%s\n" % item)
with open('Lind/im_rho_01.txt', 'w') as f:
    for item in im_rho[0][1]:
        f.write("%s\n" % item)
with open('Lind/im_rho_10.txt', 'w') as f:
    for item in im_rho[1][0]:
        f.write("%s\n" % item)

# Plotting
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
plt.title(r'$\omega$=3, $\Gamma$=1, $n$=10', fontsize=20.0)
plt.show()
