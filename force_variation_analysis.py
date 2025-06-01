import numpy as np
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.optimize as opt

# Import the core system from SystemB
# Constants
m1, m2, m3 = 1, 2, 0.5
L1, L2 = 0.5, 1.5
I1, I2 = 0.5, 1.5
I_flywheel = 1.0
F0_amplitude = 50
k = 1
g = 9.81

# Function for force modulation
def calculate_force_modulation_factor(time, mode, amplitude, frequency):
    scaled_time = frequency * time
    match mode:
        case "current_sin":
            factor = amplitude * np.sin(scaled_time)
        case "sin":
            factor = amplitude * np.sin(scaled_time)
        case "constant":
            factor = amplitude 
        case _:
            factor = amplitude * np.sin(scaled_time)
    return factor

# Set up symbolic system (simplified version of SystemB)
t = sp.symbols('t')
current_F0_base_magnitude = sp.symbols('current_F0_base_magnitude')
x1, x2, y1, y2, theta1, theta2, x3, y3 = dynamicsymbols('x1 x2 y1 y2 theta1 theta2 x3 y3')
q = sp.Matrix([x1, y1, theta1, x2, y2, theta2, x3, y3])
dq = q.diff(t)

x_com_1 = sp.Matrix([x1, y1])
x_com_2 = sp.Matrix([x2, y2])
x_com_3 = sp.Matrix([x3, y3])

R = lambda theta: sp.Matrix([[sp.cos(theta), -sp.sin(theta)], [sp.sin(theta), sp.cos(theta)]])

M = np.diag([m1, m1, I1 + I_flywheel, m2, m2, I2, m3, m3])
W = np.linalg.inv(M)
Q = sp.Matrix([0, -m1*g, -k*theta1.diff(t), 0, -m2*g, 0, 0, -m3*g + current_F0_base_magnitude * sp.cos(theta1)])

# Constraints
i_cap = sp.Matrix([1, 0])
j_cap = sp.Matrix([0, 1])

constraint_1 = x_com_1 + R(theta1) @ sp.Matrix([-L1/2, 0])
C1 = constraint_1.dot(i_cap)
C2 = constraint_1.dot(j_cap)

constraint_2 = x_com_1 - x_com_2 + R(theta1) @ sp.Matrix([L1/2, 0]) - R(theta2) @ sp.Matrix([-L2/2, 0])
C3 = constraint_2.dot(i_cap)
C4 = constraint_2.dot(j_cap)

constraint_3 = x_com_2 + R(theta2) @ sp.Matrix([L2/2, 0]) - x_com_3
C5 = constraint_3.dot(i_cap)
C6 = constraint_3.dot(j_cap)

constraint_4 = x_com_3[0]
C7 = constraint_4

C = sp.Matrix([C1, C2, C3, C4, C5, C6, C7])

# System matrices
J = C.jacobian(q)     
dq = q.diff(t)        
dC = J @ dq
dJ = dC.jacobian(q)
JWJT = J @ W @ J.T
RHS = -dJ @ dq - J @ W @ Q - 1 * C - 1 * dC

# Lambdify functions
JWJT_fn = sp.lambdify(args=(q, dq), expr=JWJT)
RHS_fn = sp.lambdify(args=(q, dq, current_F0_base_magnitude), expr=RHS)
C_fn = sp.lambdify(args=(q, dq), expr=C)    
J_fn = sp.lambdify(args=(q, dq), expr=J)   
dC_fn = sp.lambdify(args=(q, dq), expr=dC)  
dJ_fn = sp.lambdify(args=(q, dq), expr=dJ)
Q_fn = sp.lambdify(args=(q, dq, current_F0_base_magnitude), expr=Q)

# Initial conditions
dtheta1 = 0.5
initial_position_body_1 = np.array([0, L1/2, np.pi/2])
initial_position_body_2 = np.array([0, L1 + L2/2, np.pi/2])
initial_position_body_3 = np.array([0, L1 + L2])
initial_velocity_body_1 = np.array([0, 0, dtheta1])
initial_velocity_body_2 = np.array([0, 0, 0])
initial_velocity_body_3 = np.array([0, 0])
x0_temp = np.concatenate((initial_position_body_1, initial_position_body_2, initial_position_body_3,
                    initial_velocity_body_1, initial_velocity_body_2, initial_velocity_body_3))

# Calculate consistent initial conditions
x, _ = np.split(x0_temp, 2)
def optimiser(b):
    dx1, dy1, dx2, dy2, dtheta2, dx3, dy3 = b
    dq_opt = np.array([dx1, dy1, dtheta1, dx2, dy2, dtheta2, dx3, dy3])
    val = dC_fn(x, dq_opt).flatten()
    return val

initial_guess = np.array([0, 0, 0, 0, 0, 0, 0])
result = opt.root(optimiser, initial_guess)
b = result.x
dx = np.array([b[0], b[1], dtheta1, b[2], b[3], b[4], b[5], b[6]])
x0 = np.concatenate((x, dx))

def piston_engine(t, state, force_modulation_amplitude, force_variation_frequency, F0_MODULATION_MODE):
    q, dq = np.split(state, 2)
    
    modulation_factor = calculate_force_modulation_factor(time=t, 
                                                          mode=F0_MODULATION_MODE, 
                                                          amplitude=force_modulation_amplitude, 
                                                          frequency=force_variation_frequency)
    actual_F0_base_magnitude = F0_amplitude * (1 + modulation_factor)
    
    lam = np.linalg.solve(JWJT_fn(q,dq), RHS_fn(q, dq, actual_F0_base_magnitude))
    Qhat = J_fn(q, dq).T @ lam
    ddq = W @ (Q_fn(q, dq, actual_F0_base_magnitude) + Qhat)
    ddq = ddq.flatten()
    
    return np.concatenate((dq, ddq))

# Function to run simulation with given parameters
def run_simulation(force_mod_amp, force_var_freq, mode, t_span=(0, 20), n_points=400):
    t_eval = np.linspace(*t_span, n_points)
    
    def system_wrapper(t, state):
        return piston_engine(t, state, force_mod_amp, force_var_freq, mode)
    
    sol = solve_ivp(system_wrapper, t_span, x0, atol=1e-7, rtol=1e-7, method='BDF', t_eval=t_eval)
    return sol

print("Starting force variation analysis...")

# 1. Compare constant vs time-varying force
print("1. Generating constant vs time-varying force comparison...")

# Case 1: Constant force (A_m = 0)
sol_constant = run_simulation(force_mod_amp=0.0, force_var_freq=1.0, mode="constant")

# Case 2: Time-varying force (A_m = 0.5, sinusoidal)
sol_varying = run_simulation(force_mod_amp=0.5, force_var_freq=1.0, mode="sin")

# Extract angular velocities
theta1_dot_constant = sol_constant.y[10]  # dtheta1/dt
theta1_dot_varying = sol_varying.y[10]    # dtheta1/dt

# Extract positions for torque calculation
theta1_constant = sol_constant.y[2]
theta1_varying = sol_varying.y[2]

# Calculate applied forces over time
def calculate_applied_force(t, theta1, force_mod_amp, force_var_freq, mode):
    modulation_factor = calculate_force_modulation_factor(t, mode, force_mod_amp, force_var_freq)
    actual_F0 = F0_amplitude * (1 + modulation_factor)
    return actual_F0 * np.cos(theta1)

force_constant = np.array([calculate_applied_force(t, th, 0.0, 1.0, "constant") 
                          for t, th in zip(sol_constant.t, theta1_constant)])
force_varying = np.array([calculate_applied_force(t, th, 0.5, 1.0, "sin") 
                         for t, th in zip(sol_varying.t, theta1_varying)])

# Plot comparison
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Angular velocity comparison
ax1.plot(sol_constant.t, theta1_dot_constant, 'b-', label='Constant Force (A_m = 0)', linewidth=2)
ax1.plot(sol_varying.t, theta1_dot_varying, 'r-', label='Time-varying Force (A_m = 0.5)', linewidth=2)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Angular Velocity (rad/s)')
ax1.set_title('Crankshaft Angular Velocity Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Applied force comparison
ax2.plot(sol_constant.t, force_constant, 'b-', label='Constant Force', linewidth=2)
ax2.plot(sol_varying.t, force_varying, 'r-', label='Time-varying Force', linewidth=2)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Applied Force (N)')
ax2.set_title('Applied Piston Force Comparison')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Crank angle comparison
ax3.plot(sol_constant.t, theta1_constant, 'b-', label='Constant Force', linewidth=2)
ax3.plot(sol_varying.t, theta1_varying, 'r-', label='Time-varying Force', linewidth=2)
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Crank Angle (rad)')
ax3.set_title('Crank Angle vs Time')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Force modulation factor over time
t_mod = np.linspace(0, 20, 400)
modulation_factors = [calculate_force_modulation_factor(t, "sin", 0.5, 1.0) for t in t_mod]
actual_F0_values = F0_amplitude * (1 + np.array(modulation_factors))

ax4.plot(t_mod, actual_F0_values, 'g-', linewidth=2)
ax4.axhline(y=F0_amplitude, color='b', linestyle='--', label='Base F0')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('F0 Magnitude (N)')
ax4.set_title('Time-varying F0 Amplitude')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('force_variation_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Force variation comparison plot saved as 'force_variation_comparison.png'")

# 2. Different modulation amplitudes
print("2. Generating different modulation amplitude comparison...")

amplitudes = [0.0, 0.25, 0.5, 0.75]
colors = ['blue', 'green', 'red', 'purple']

plt.figure(figsize=(12, 8))

for i, amp in enumerate(amplitudes):
    sol = run_simulation(force_mod_amp=amp, force_var_freq=1.0, mode="sin")
    theta1_dot = sol.y[10]
    
    plt.subplot(2, 1, 1)
    plt.plot(sol.t, theta1_dot, color=colors[i], 
             label=f'A_m = {amp}', linewidth=2)

plt.subplot(2, 1, 1)
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.title('Effect of Force Modulation Amplitude on Angular Velocity')
plt.legend()
plt.grid(True, alpha=0.3)

# Show the corresponding force modulation
t_range = np.linspace(0, 20, 400)
for i, amp in enumerate(amplitudes):
    modulation = [calculate_force_modulation_factor(t, "sin", amp, 1.0) for t in t_range]
    F0_values = F0_amplitude * (1 + np.array(modulation))
    
    plt.subplot(2, 1, 2)
    plt.plot(t_range, F0_values, color=colors[i], 
             label=f'A_m = {amp}', linewidth=2)

plt.subplot(2, 1, 2)
plt.xlabel('Time (s)')
plt.ylabel('F0 Amplitude (N)')
plt.title('Force Modulation for Different Amplitudes')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('force_modulation_amplitudes.png', dpi=300, bbox_inches='tight')
plt.show()

print("Force modulation amplitudes plot saved as 'force_modulation_amplitudes.png'")

print("Force variation analysis complete!") 