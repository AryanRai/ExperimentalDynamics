import numpy as np
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.optimize as opt

# Constants (same as SystemB)
m1, m2, m3 = 1, 2, 0.5
L1, L2 = 0.5, 1.5
I1, I2 = 0.5, 1.5
F0_amplitude = 50
k = 1
g = 9.81

# Force modulation function (simplified)
def calculate_force_modulation_factor(time, mode, amplitude, frequency):
    if mode == "constant":
        return 0
    else:
        return amplitude * np.sin(frequency * time)

# Create function to build system with variable flywheel inertia
def create_system(I_flywheel):
    # Set up symbolic system
    t = sp.symbols('t')
    current_F0_base_magnitude = sp.symbols('current_F0_base_magnitude')
    x1, x2, y1, y2, theta1, theta2, x3, y3 = dynamicsymbols('x1 x2 y1 y2 theta1 theta2 x3 y3')
    q = sp.Matrix([x1, y1, theta1, x2, y2, theta2, x3, y3])
    dq = q.diff(t)

    x_com_1 = sp.Matrix([x1, y1])
    x_com_2 = sp.Matrix([x2, y2])
    x_com_3 = sp.Matrix([x3, y3])

    R = lambda theta: sp.Matrix([[sp.cos(theta), -sp.sin(theta)], [sp.sin(theta), sp.cos(theta)]])

    # Include flywheel inertia in the mass matrix
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
    Q_fn = sp.lambdify(args=(q, dq, current_F0_base_magnitude), expr=Q)
    
    return JWJT_fn, RHS_fn, C_fn, J_fn, dC_fn, Q_fn, W

# Function to get initial conditions
def get_initial_conditions(system_funcs):
    JWJT_fn, RHS_fn, C_fn, J_fn, dC_fn, Q_fn, W = system_funcs
    
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
    
    return x0

# Function to create piston engine function for given system
def create_piston_engine(system_funcs):
    JWJT_fn, RHS_fn, C_fn, J_fn, dC_fn, Q_fn, W = system_funcs
    
    def piston_engine(t, state):
        q, dq = np.split(state, 2)
        
        # Use constant force for flywheel comparison
        actual_F0_base_magnitude = F0_amplitude
        
        lam = np.linalg.solve(JWJT_fn(q,dq), RHS_fn(q, dq, actual_F0_base_magnitude))
        Qhat = J_fn(q, dq).T @ lam
        ddq = W @ (Q_fn(q, dq, actual_F0_base_magnitude) + Qhat)
        ddq = ddq.flatten()
        
        return np.concatenate((dq, ddq))
    
    return piston_engine

# Function to run simulation with given flywheel inertia
def run_flywheel_simulation(I_flywheel, t_span=(0, 30), n_points=500):
    system_funcs = create_system(I_flywheel)
    x0 = get_initial_conditions(system_funcs)
    piston_engine = create_piston_engine(system_funcs)
    
    t_eval = np.linspace(*t_span, n_points)
    sol = solve_ivp(piston_engine, t_span, x0, atol=1e-7, rtol=1e-7, method='BDF', t_eval=t_eval)
    return sol

print("Starting flywheel analysis...")

# 1. Compare with and without flywheel
print("1. Generating with/without flywheel comparison...")

# Case 1: No flywheel (I_flywheel = 0)
sol_no_flywheel = run_flywheel_simulation(I_flywheel=0.0)

# Case 2: With flywheel (I_flywheel = 1.0)
sol_with_flywheel = run_flywheel_simulation(I_flywheel=1.0)

# Extract data
theta1_dot_no_fly = sol_no_flywheel.y[10]  # dtheta1/dt
theta1_dot_with_fly = sol_with_flywheel.y[10]

theta1_no_fly = sol_no_flywheel.y[2]  # theta1
theta1_with_fly = sol_with_flywheel.y[2]

y3_no_fly = sol_no_flywheel.y[7]  # piston position
y3_with_fly = sol_with_flywheel.y[7]

# Calculate angular velocity statistics
std_no_fly = np.std(theta1_dot_no_fly)
std_with_fly = np.std(theta1_dot_with_fly)

print(f"Angular velocity standard deviation:")
print(f"  Without flywheel: {std_no_fly:.4f} rad/s")
print(f"  With flywheel: {std_with_fly:.4f} rad/s")
print(f"  Reduction: {((std_no_fly - std_with_fly) / std_no_fly * 100):.1f}%")

# Create comprehensive comparison plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Angular velocity comparison
ax1.plot(sol_no_flywheel.t, theta1_dot_no_fly, 'r-', label='No Flywheel (I_f = 0)', linewidth=2)
ax1.plot(sol_with_flywheel.t, theta1_dot_with_fly, 'b-', label='With Flywheel (I_f = 1.0)', linewidth=2)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Angular Velocity (rad/s)')
ax1.set_title('Crankshaft Angular Velocity: Flywheel Impact')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Zoom in on first few seconds for better detail
ax2.plot(sol_no_flywheel.t[:100], theta1_dot_no_fly[:100], 'r-', label='No Flywheel', linewidth=2)
ax2.plot(sol_with_flywheel.t[:100], theta1_dot_with_fly[:100], 'b-', label='With Flywheel', linewidth=2)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Angular Velocity (rad/s)')
ax2.set_title('Angular Velocity Detail (First 6 seconds)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Crank angle comparison
ax3.plot(sol_no_flywheel.t, theta1_no_fly, 'r-', label='No Flywheel', linewidth=2)
ax3.plot(sol_with_flywheel.t, theta1_with_fly, 'b-', label='With Flywheel', linewidth=2)
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Crank Angle (rad)')
ax3.set_title('Crank Angle vs Time')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Piston displacement comparison
ax4.plot(sol_no_flywheel.t, y3_no_fly, 'r-', label='No Flywheel', linewidth=2)
ax4.plot(sol_with_flywheel.t, y3_with_fly, 'b-', label='With Flywheel', linewidth=2)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Piston Position (m)')
ax4.set_title('Piston Displacement')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('flywheel_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Flywheel comparison plot saved as 'flywheel_comparison.png'")

# 2. Different flywheel inertias
print("2. Generating different flywheel inertia comparison...")

inertias = [0.0, 0.5, 1.0, 2.0]
colors = ['red', 'orange', 'blue', 'green']
std_values = []

plt.figure(figsize=(12, 10))

for i, I_f in enumerate(inertias):
    print(f"  Simulating I_flywheel = {I_f}...")
    sol = run_flywheel_simulation(I_flywheel=I_f)
    theta1_dot = sol.y[10]
    std_val = np.std(theta1_dot)
    std_values.append(std_val)
    
    # Plot angular velocity
    plt.subplot(2, 1, 1)
    plt.plot(sol.t, theta1_dot, color=colors[i], 
             label=f'I_f = {I_f} (σ = {std_val:.3f})', linewidth=2)

plt.subplot(2, 1, 1)
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.title('Effect of Flywheel Inertia on Angular Velocity')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot standard deviation vs flywheel inertia
plt.subplot(2, 1, 2)
plt.plot(inertias, std_values, 'o-', color='purple', linewidth=2, markersize=8)
plt.xlabel('Flywheel Inertia (kg⋅m²)')
plt.ylabel('Angular Velocity Std Dev (rad/s)')
plt.title('Angular Velocity Stability vs Flywheel Inertia')
plt.grid(True, alpha=0.3)

# Add annotation for improvement
for i, (I_f, std_val) in enumerate(zip(inertias, std_values)):
    plt.annotate(f'{std_val:.3f}', (I_f, std_val), 
                textcoords="offset points", xytext=(0,10), ha='center')

plt.tight_layout()
plt.savefig('flywheel_inertia_effects.png', dpi=300, bbox_inches='tight')
plt.show()

print("Flywheel inertia effects plot saved as 'flywheel_inertia_effects.png'")

# 3. Torque smoothing analysis
print("3. Generating torque smoothing analysis...")

# Calculate generalized forces (torques) for comparison
sol_no_fly = run_flywheel_simulation(I_flywheel=0.0)
sol_with_fly = run_flywheel_simulation(I_flywheel=1.0)

# Calculate applied torque from piston force
def calculate_piston_torque(theta1_values):
    """Calculate torque applied by piston force"""
    torque = F0_amplitude * np.cos(theta1_values) * L1 * np.sin(theta1_values)
    return torque

torque_no_fly = calculate_piston_torque(sol_no_fly.y[2])
torque_with_fly = calculate_piston_torque(sol_with_fly.y[2])

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(sol_no_fly.t, torque_no_fly, 'r-', label='No Flywheel', linewidth=2)
plt.plot(sol_with_fly.t, torque_with_fly, 'b-', label='With Flywheel', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Piston Torque (N⋅m)')
plt.title('Piston Torque Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# Show detail for first few cycles
plt.subplot(2, 1, 2)
t_detail = sol_no_fly.t[:150]
plt.plot(t_detail, torque_no_fly[:150], 'r-', label='No Flywheel', linewidth=2)
plt.plot(t_detail, torque_with_fly[:150], 'b-', label='With Flywheel', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Piston Torque (N⋅m)')
plt.title('Piston Torque Detail (First 9 seconds)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('torque_smoothing_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("Torque smoothing analysis plot saved as 'torque_smoothing_analysis.png'")

# Summary statistics
print("\n" + "="*50)
print("FLYWHEEL ANALYSIS SUMMARY")
print("="*50)
print(f"Angular Velocity Standard Deviation:")
for I_f, std_val in zip(inertias, std_values):
    reduction = ((std_values[0] - std_val) / std_values[0] * 100) if I_f > 0 else 0
    print(f"  I_f = {I_f:3.1f}: σ = {std_val:.4f} rad/s ({reduction:5.1f}% reduction)")

print(f"\nTorque Standard Deviation:")
torque_std_no_fly = np.std(torque_no_fly)
torque_std_with_fly = np.std(torque_with_fly)
torque_reduction = ((torque_std_no_fly - torque_std_with_fly) / torque_std_no_fly * 100)
print(f"  No flywheel:  σ = {torque_std_no_fly:.4f} N⋅m")
print(f"  With flywheel: σ = {torque_std_with_fly:.4f} N⋅m ({torque_reduction:.1f}% reduction)")

print("\nFlywheel analysis complete!") 