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

# Test
plt.plot(time, val_1, 'k-', linewidth=2.0, label='')
plt.plot(time, val_2, 'b-', linewidth=2.0, label='')
plt.plot(time, val_3, 'r-', linewidth=2.0, label='')
plt.xlim([0.0,7.0])
plt.show()
