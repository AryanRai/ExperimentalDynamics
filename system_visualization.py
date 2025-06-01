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

# Force modulation function
def calculate_force_modulation_factor(time, mode, amplitude, frequency):
    if mode == "constant":
        return 0
    else:
        return amplitude * np.sin(frequency * time)

# System setup function
def create_system(I_flywheel, force_modulation_amplitude=0.0, force_variation_frequency=1.0, F0_MODULATION_MODE="constant"):
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
    
    def piston_engine(t, state):
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
    
    return piston_engine, C_fn, dC_fn, W

# Get initial conditions
def get_initial_conditions(system_components):
    piston_engine, C_fn, dC_fn, W = system_components
    
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

# Visualization classes (from SystemB)
class Box:
    def __init__(self, width, height, color='b'):
        self.width = width
        self.height = height
        self.color = color
        self.offset = -np.array([width/2, height/2])

    def draw_static(self, ax, x, y, theta):
        theta_deg = np.rad2deg(theta)
        corner = np.array([x, y]) + self.offset
        rect = plt.Rectangle(corner, self.width, self.height, angle=theta_deg, 
                            rotation_point='center', color=self.color, alpha=0.8)
        ax.add_patch(rect)
        return rect

class CircleBody:
    def __init__(self, radius, color='purple'):
        self.radius = radius
        self.color = color

    def draw_static(self, ax, x, y, theta=0):
        circle = plt.Circle((x, y), self.radius, color=self.color, alpha=0.6, zorder=-1)
        ax.add_patch(circle)
        return circle

# Function to create system snapshot
def create_system_snapshot(I_flywheel, force_params, title, filename, time_point=0.5):
    """Create a static snapshot of the system at a specific time point"""
    
    force_modulation_amplitude, force_variation_frequency, F0_MODULATION_MODE = force_params
    
    # Create system
    system_components = create_system(I_flywheel, force_modulation_amplitude, 
                                    force_variation_frequency, F0_MODULATION_MODE)
    piston_engine, C_fn, dC_fn, W = system_components
    x0 = get_initial_conditions(system_components)
    
    # Run short simulation to get to desired time point
    t_span = (0, time_point + 0.1)
    t_eval = np.linspace(*t_span, 50)
    sol = solve_ivp(piston_engine, t_span, x0, atol=1e-7, rtol=1e-7, method='BDF', t_eval=t_eval)
    
    # Find closest time point
    closest_idx = np.argmin(np.abs(sol.t - time_point))
    
    # Extract positions
    x1, y1, theta1 = sol.y[0, closest_idx], sol.y[1, closest_idx], sol.y[2, closest_idx]
    x2, y2, theta2 = sol.y[3, closest_idx], sol.y[4, closest_idx], sol.y[5, closest_idx]
    x3, y3 = sol.y[6, closest_idx], sol.y[7, closest_idx]
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.8, 2.3)
    ax.set_aspect('equal')
    
    # Draw components
    if I_flywheel > 0:
        flywheel = CircleBody(radius=L1*0.8, color='purple')
        flywheel.draw_static(ax, 0, 0)
        ax.text(0, -L1*0.8-0.1, f'Flywheel\nI = {I_flywheel} kg⋅m²', 
                ha='center', va='top', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lavender"))
    
    # Draw crank (body 1)
    crank = Box(L1, 0.05, 'cyan')
    crank.draw_static(ax, x1, y1, theta1)
    
    # Draw connecting rod (body 2)
    conrod = Box(L2, 0.03, 'red')
    conrod.draw_static(ax, x2, y2, theta2)
    
    # Draw piston (body 3)
    piston = Box(0.15, 0.3, 'green')
    piston.draw_static(ax, x3, y3, 0)
    
    # Draw ground and guides
    ax.axhline(y=0, color='black', linewidth=3, label='Ground')
    ax.axvline(x=-0.1, color='gray', linewidth=2, linestyle='--', alpha=0.7)
    ax.axvline(x=0.1, color='gray', linewidth=2, linestyle='--', alpha=0.7)
    
    # Add labels and annotations
    ax.plot(0, 0, 'ko', markersize=8, label='Fixed Pivot')
    
    # Force arrow on piston
    force_magnitude = F0_amplitude
    if force_modulation_amplitude > 0:
        modulation = calculate_force_modulation_factor(time_point, F0_MODULATION_MODE, 
                                                     force_modulation_amplitude, force_variation_frequency)
        force_magnitude = F0_amplitude * (1 + modulation)
    
    actual_force = force_magnitude * np.cos(theta1)
    arrow_length = abs(actual_force) / 50  # Scale for visualization
    
    if actual_force > 0:
        ax.arrow(x3, y3 + 0.2, 0, arrow_length, head_width=0.05, head_length=0.05, 
                fc='orange', ec='orange', linewidth=2)
        ax.text(x3 + 0.2, y3 + 0.2, f'F = {actual_force:.1f} N', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat"))
    
    # Add dimension annotations
    ax.annotate('', xy=(0, 0), xytext=(x1, y1), 
                arrowprops=dict(arrowstyle='<->', color='blue', lw=1))
    ax.text((x1)/2, (y1)/2 + 0.1, f'L₁ = {L1} m', fontsize=9, ha='center', color='blue')
    
    ax.annotate('', xy=(x1, y1), xytext=(x3, y3), 
                arrowprops=dict(arrowstyle='<->', color='red', lw=1))
    mid_x, mid_y = (x1 + x3)/2, (y1 + y3)/2
    ax.text(mid_x + 0.15, mid_y, f'L₂ = {L2} m', fontsize=9, ha='center', color='red')
    
    # Title and parameters
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Parameter box
    param_text = f"Parameters:\n"
    param_text += f"t = {time_point:.1f} s\n"
    param_text += f"θ₁ = {np.rad2deg(theta1):.1f}°\n"
    param_text += f"F₀ = {F0_amplitude} N\n"
    if force_modulation_amplitude > 0:
        param_text += f"A_m = {force_modulation_amplitude}\n"
        param_text += f"ω_f = {force_variation_frequency} rad/s"
    
    ax.text(0.02, 0.98, param_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Position (m)')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

# Function to create multi-snapshot comparison
def create_system_comparison(scenarios, filename, times=[0.5, 1.0]):
    """Create side-by-side comparison of different system configurations"""
    
    n_scenarios = len(scenarios)
    n_times = len(times)
    
    fig, axes = plt.subplots(n_times, n_scenarios, figsize=(4*n_scenarios, 6*n_times))
    if n_times == 1:
        axes = axes.reshape(1, -1)
    if n_scenarios == 1:
        axes = axes.reshape(-1, 1)
    
    for t_idx, time_point in enumerate(times):
        for s_idx, (title, I_flywheel, force_params) in enumerate(scenarios):
            ax = axes[t_idx, s_idx]
            
            force_modulation_amplitude, force_variation_frequency, F0_MODULATION_MODE = force_params
            
            # Create system and simulate
            system_components = create_system(I_flywheel, force_modulation_amplitude, 
                                            force_variation_frequency, F0_MODULATION_MODE)
            piston_engine, C_fn, dC_fn, W = system_components
            x0 = get_initial_conditions(system_components)
            
            t_span = (0, time_point + 0.1)
            t_eval = np.linspace(*t_span, 50)
            sol = solve_ivp(piston_engine, t_span, x0, atol=1e-7, rtol=1e-7, method='BDF', t_eval=t_eval)
            
            closest_idx = np.argmin(np.abs(sol.t - time_point))
            x1, y1, theta1 = sol.y[0, closest_idx], sol.y[1, closest_idx], sol.y[2, closest_idx]
            x2, y2, theta2 = sol.y[3, closest_idx], sol.y[4, closest_idx], sol.y[5, closest_idx]
            x3, y3 = sol.y[6, closest_idx], sol.y[7, closest_idx]
            
            # Draw system
            ax.set_xlim(-0.8, 0.8)
            ax.set_ylim(-0.8, 2.3)
            ax.set_aspect('equal')
            
            # Draw flywheel if present
            if I_flywheel > 0:
                flywheel = CircleBody(radius=L1*0.8, color='purple')
                flywheel.draw_static(ax, 0, 0)
            
            # Draw components
            crank = Box(L1, 0.05, 'cyan')
            crank.draw_static(ax, x1, y1, theta1)
            
            conrod = Box(L2, 0.03, 'red')
            conrod.draw_static(ax, x2, y2, theta2)
            
            piston = Box(0.15, 0.3, 'green')
            piston.draw_static(ax, x3, y3, 0)
            
            # Ground and guides
            ax.axhline(y=0, color='black', linewidth=2)
            ax.axvline(x=-0.1, color='gray', linewidth=1, linestyle='--', alpha=0.7)
            ax.axvline(x=0.1, color='gray', linewidth=1, linestyle='--', alpha=0.7)
            ax.plot(0, 0, 'ko', markersize=6)
            
            # Title
            if t_idx == 0:
                ax.set_title(title, fontsize=12, fontweight='bold')
            
            # Time label
            if s_idx == 0:
                ax.set_ylabel(f't = {time_point:.1f} s', fontsize=11, fontweight='bold')
            
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Position (m)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

print("Starting system visualization generation...")

# 1. Flywheel comparison snapshots
print("1. Creating flywheel comparison images...")

flywheel_scenarios = [
    ("Without Flywheel", 0.0, (0.0, 1.0, "constant")),
    ("With Flywheel (I = 1.0)", 1.0, (0.0, 1.0, "constant")),
]

create_system_comparison(flywheel_scenarios, "system_flywheel_comparison.png", times=[1.0, 2.0])

# 2. Force variation comparison snapshots  
print("2. Creating force variation comparison images...")

force_scenarios = [
    ("Constant Force", 1.0, (0.0, 1.0, "constant")),
    ("Time-varying Force", 1.0, (0.5, 1.0, "sin")),
]

create_system_comparison(force_scenarios, "system_force_comparison.png", times=[1.0, 2.5])

# 3. Individual detailed snapshots
print("3. Creating detailed individual system snapshots...")

# High-quality individual snapshots
create_system_snapshot(0.0, (0.0, 1.0, "constant"), 
                      "Reciprocating Piston System - No Flywheel", 
                      "system_no_flywheel.png", time_point=1.0)

create_system_snapshot(1.0, (0.0, 1.0, "constant"), 
                      "Reciprocating Piston System - With Flywheel", 
                      "system_with_flywheel.png", time_point=1.0)

create_system_snapshot(1.0, (0.5, 1.0, "sin"), 
                      "Reciprocating Piston System - Time-varying Force", 
                      "system_time_varying_force.png", time_point=2.0)

# 4. System at different positions
print("4. Creating system position sequence...")

position_scenarios = [
    ("Top Dead Center", 1.0, (0.0, 1.0, "constant")),
    ("Mid-stroke", 1.0, (0.0, 1.0, "constant")),
    ("Bottom Dead Center", 1.0, (0.0, 1.0, "constant")),
]

create_system_comparison(position_scenarios, "system_position_sequence.png", 
                        times=[0.0, 0.8, 1.57])  # Different crank positions

print("System visualization generation complete!")
print("\nGenerated files:")
print("- system_flywheel_comparison.png")
print("- system_force_comparison.png") 
print("- system_no_flywheel.png")
print("- system_with_flywheel.png")
print("- system_time_varying_force.png")
print("- system_position_sequence.png") 