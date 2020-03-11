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
# for an arbitrarly large matrix sparse initialization is better
Gamma = np.zeros( (4, 4), dtype=float)
Gamma[0,0] = -w_41
Gamma[0,1] = w_12
Gamma[1,1] = -w_12
Gamma[1,2] = w_23
Gamma[2,2] = -w_23
Gamma[2,3] = w_34
Gamma[3,0] = w_41
Gamma[3,3] = -w_34

# Eigenvalues/eigenvectors
# the minimum eigenvale is set to zero to avoid numerical issues
Lambda, V = la.eig(Gamma)
Lambda[0] = 0.0

# Finding coefficients
p_0 = np.array([1.0, 0.0, 0.0, 0.0])
c = la.solve(V, p_0)

# Evaluating probabilities and time constant
idx = np.nonzero( np.abs( np.real(Lambda) ) )
tau_max = 1.0 / np.min( np.abs( np.real(Lambda[idx]) ) )
t = np.linspace(0, 7*tau_max, 250)
exp_l1t = np.exp( Lambda[1]*t )
exp_l2t = np.exp( Lambda[2]*t )
exp_l3t = np.exp( Lambda[3]*t )

p_1 = c[0]*V[0,0] + c[1]*exp_l1t*V[0,1] + \
    c[2]*exp_l2t*V[0,2] + c[3]*exp_l3t*V[0,3]
p_2 = c[0]*V[1,0] + c[1]*exp_l1t*V[1,1] + \
    c[2]*exp_l2t*V[1,2] + c[3]*exp_l3t*V[1,3]
p_3 = c[0]*V[2,0] + c[1]*exp_l1t*V[2,1] + \
    c[2]*exp_l2t*V[2,2] + c[3]*exp_l3t*V[2,3]
p_4 = c[0]*V[3,0] + c[1]*exp_l1t*V[3,1] + \
    c[2]*exp_l2t*V[3,2] + c[3]*exp_l3t*V[3,3]

sum_p = p_1+p_2+p_3+p_4

# Testing if the final state is indeed a vector of the
# null space of Gamma
p_inf = np.array( [p_1[-1], p_2[-1], p_3[-1], p_4[-1]] )
residue = np.dot(Gamma, p_inf)

# Print output
print("Final state:\n"+str(p_inf))
print("Residue:\n"+str(residue))

# Plotting
plt.plot(t, p_1, 'k-', label='P1')
plt.plot(t, p_2, 'b-', label='P2')
plt.plot(t, p_3, 'r-', label='P3')
plt.plot(t, p_4, 'g-', label='P4')
plt.plot(t, sum_p, 'k--', label='Sum')
plt.xlim([0, max(t)])
plt.xlabel('t', fontsize=20)
plt.ylabel('P(t)', fontsize=20)
plt.title('Master equation (probabilities)', fontsize=20)
plt.legend(prop=dict(size=18))
plt.show()
