from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

K = 1000
time = np.linspace(0.0, 7.0, K)

def kernel(t, oc) :
    if t>0 :
        return (2.0/np.pi)*np.sin(oc*t)/t
    elif t==0 :
        return (2.0/np.pi)*oc
    else :
        return 0

# def kernel(t) :
#     if t==0 :
#         return (1.0/np.pi)*omega_cut
#     else :
#         return (1.0/np.pi)*np.sin(omega_cut*t)/t

omega_cut_1 = 10.0
omega_cut_2 = 5.0
omega_cut_3 = 2.0
val_1 = np.zeros(K, dtype=float)
val_2 = np.zeros(K, dtype=float)
val_3 = np.zeros(K, dtype=float)

for k in range(K):
    val_1[k] = kernel(time[k], omega_cut_1)
    val_2[k] = kernel(time[k], omega_cut_2)
    val_3[k] = kernel(time[k], omega_cut_3)

"""
plt.plot(time, val_1, 'k-', linewidth=2.0, label='')
plt.plot(time, val_2, 'b-', linewidth=2.0, label='')
plt.plot(time, val_3, 'r-', linewidth=2.0, label='')
plt.xlim([0.0,7.0])
plt.show()
"""

# Lennard Jones

LJ = lambda r : (1.0/r)**12 - 2.0*(1.0/r)**6
d2_LJ_d_r2 = lambda r : 12 * ( 13*(1.0/r)**14 - 7*(1.0/r)**8 )

c = -1.0
a = d2_LJ_d_r2(1.0)

r_vec = np.linspace(0.5, 2.0, 1000)
r_qua = np.linspace(0.0, 2.0, 1000)

plt.plot(r_vec, LJ(r_vec), 'k-', label='Lennard Jones')
plt.plot(r_qua, a*(r_qua-1.0)**2+c, 'r-', label='Quadratic' )
plt.ylim([-1.5, 1.25])
plt.xlim([0.5,1.5])
plt.xlabel('x')
plt.ylabel('V')
plt.legend(loc='lower left')
plt.show()
