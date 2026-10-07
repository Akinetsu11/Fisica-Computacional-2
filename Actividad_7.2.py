import numpy as np
from numpy import cos, pi
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# System parameters (Ejercicio 2)
# -----------------------------------------------------------

q = 2.0
gamma = 1.1799
omega = 2/3
T = 2 * pi / omega         # Periodo de la fuerza externa

# -----------------------------------------------------------
# Initial conditions: grid adaptado a la Fig. 1-(a)
# -----------------------------------------------------------
x0_values = np.linspace(-2.0, 2.0, 50)
v0_values = np.linspace(-pi, pi, 20)
X0, V0 = np.meshgrid(x0_values, v0_values)
x0_flat = X0.ravel()
v0_flat = V0.ravel()
n_orbits = len(x0_flat)
print(f"Número de órbitas: {n_orbits}")

y = np.concatenate([x0_flat, v0_flat])

# -----------------------------------------------------------
# Method parameters
# -----------------------------------------------------------
Trans = 10    # Períodos transitorios a descartar
Nperiods = 100      # Períodos totales de simulación
steps_per_T = 100    # Resolución de pasos por período T
dt = T / steps_per_T

# -----------------------------------------------------------
# Dynamics: Forced Duffing oscillator
# -----------------------------------------------------------
def dyn(t, y_state):
    x = y_state[:n_orbits]
    v = y_state[n_orbits:]
    dx = v
    dv = -(1/q) * v -(np.sin(x)) + gamma * cos(omega * t)
    return np.concatenate([dx, dv])

# -----------------------------------------------------------
# Fourth-order Runge-Kutta method
# -----------------------------------------------------------
def rk4(f, t, y_state, h):
    k1 = h * f(t, y_state)
    k2 = h * f(t + h/2, y_state + k1/2)
    k3 = h * f(t + h/2, y_state + k2/2)
    k4 = h * f(t + h, y_state + k3)
    return y_state + (k1 + 2*k2 + 2*k3 + k4) / 6

# -----------------------------------------------------------
# Stroboscopic storage setup
# -----------------------------------------------------------
def wrap(x):
    return (x+pi)%(2*pi)-pi
n_saved = Nperiods - Trans + 1
x_strobe = np.empty((n_saved, n_orbits))
v_strobe = np.empty((n_saved, n_orbits))
save_index = 0

total_steps = Nperiods * steps_per_T

# -----------------------------------------------------------
# Integration and Stroboscopic Sampling
# -----------------------------------------------------------
for step in range(total_steps):
    current_time = step * dt
    y = rk4(dyn, current_time, y, dt)
    
    completed_period = (step + 1) // steps_per_T
    if (step + 1) % steps_per_T == 0:
        if completed_period >= Trans:
            x_strobe[save_index] = wrap(y[:n_orbits])
            v_strobe[save_index] = y[n_orbits:]
            save_index += 1

# -----------------------------------------------------------
# Stroboscopic Map Plot (Ejercicio 2)
# -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4), dpi=200)

colors = ['red', 'orange', 'green', 'blue', 'purple']
for i in range(n_orbits):
    c = colors[i % len(colors)]
    ax.scatter(x_strobe[:, i], v_strobe[:, i], s=0.8, color=c, alpha=0.8, linewidths=0, rasterized=True)

ax.set_xlabel(r'$\theta$', fontsize=22)
ax.set_ylabel(r'$\dot{\theta}$', fontsize=22)
ax.tick_params(axis='both', labelsize=18)
ax.set_box_aspect(0.65)
plt.tight_layout()
plt.savefig("mapa_estroboscopico_2.png", format="png", bbox_inches="tight", dpi=300)
plt.show()
