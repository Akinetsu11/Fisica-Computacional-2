import numpy as np
import matplotlib.pyplot as plt

#vertices triangulo equilatero
V=np.array([[0,0],[0.5,0],[1,0],[0,0.5],[1,0.5],[0,1],[0.5,1],[1,1]])

#punto inicial
P=np.array([0,1])

#iteraciones
N=100000
x=np.zeros(N)
y=np.zeros(N)

#algoritmo
for k in range(N):
  i=np.random.randint(0,8)
  P=P/3 + 2*V[i]/3
  x[k]=P[0]
  y[k]=P[1]


# scatter plot
fig,ax= plt.subplots(figsize=(8,8))
ax.scatter(x,y,s=0.1)
ax.text(0.03,0.97,rf'$N={N}$',transform=ax.transAxes,ha='left',va='top',fontsize=12)
ax.tick_params(axis='both',labelsize=14)
ax.set_aspect('equal')
plt.tight_layout()

plt.show()
