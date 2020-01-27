# SOMETHING IS TERRIBLY WRONG! IT GIVES NEGATIVE PROBABILITIES!!!

import math
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# Transition weights
w_41 = 1.0
w_12 = 2.0
w_23 = 0.1
w_34 = 2.0

# Transition matrix
# For a transition of an arbitrarly large size this initialization is better
Gamma = np.zeros( (4, 4), dtype=float)
Gamma[0,0] = -w_41
Gamma[0,1] = w_12
Gamma[1,1] = -w_12
Gamma[1,2] = w_23
Gamma[2,2] = -w_23
Gamma[2,3] = w_34
Gamma[3,0] = w_41
Gamma[3,3] = -w_34
print("Gamma = \n"+str(Gamma))

# Eigenvalues/eigenvectors; the minimum eigenvale is set to zero
# to avoid numerical issues
Lambda, V = la.eig(Gamma)
Lambda[0] = 0.0
print("Lambda = \n"+str(Lambda))
print("V = \n"+str(V))

# Finding coefficients
p_0 = np.array([1.0, 0.0, 0.0, 0.0])
c = la.solve(V, p_0)
# c = np.real(c)
print("c = \n"+str(c))

# Evaluating probabilities
idx = np.nonzero( np.abs( np.real(Lambda) ) )
tau_max = 1.0 / np.min( np.abs( np.real(Lambda[idx]) ) )
t = np.linspace(0, 7*tau_max, 250)
exp_l1t = np.exp( Lambda[1]*t )
exp_l2t = np.exp( Lambda[2]*t )
exp_l3t = np.exp( Lambda[3]*t )
# exp_l1t = np.ones(len(t))
# exp_l2t = np.ones(len(t))
# exp_l3t = np.ones(len(t))

p_1 = c[0]*V[0,0] + c[1]*exp_l1t*V[0,1] + c[2]*exp_l2t*V[0,2] + c[3]*exp_l3t*V[0,3]
p_2 = c[0]*V[1,0] + c[1]*exp_l1t*V[1,1] + c[2]*exp_l2t*V[1,2] + c[3]*exp_l3t*V[1,3]
p_3 = c[0]*V[2,0] + c[1]*exp_l1t*V[2,1] + c[2]*exp_l2t*V[2,2] + c[3]*exp_l3t*V[2,3]
p_4 = c[0]*V[3,0] + c[1]*exp_l1t*V[3,1] + c[2]*exp_l2t*V[3,2] + c[3]*exp_l3t*V[3,3]

# Plotting
plt.plot(t, p_1, 'k-')
plt.plot(t, p_2, 'b-')
plt.plot(t, p_3, 'r-')
plt.plot(t, p_4, 'g-')
plt.show()
