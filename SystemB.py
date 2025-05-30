# Original Code


# %%
# Load the necessary libraries
import numpy as np
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# This is for pretty printing
import IPython.display as disp

# %% [markdown]
# Constants

# %%
m1, m2, m3 = 1, 2, 0.5
L1, L2 = 0.5, 1.5
I1, I2 = 0.5, 1.5
# Add flywheel parameters
m_flywheel = 2.0  # Mass of the flywheel
I_flywheel = 1.0  # Moment of inertia of the flywheel

# Parameters for time-varying F0
F0_amplitude = 50  # Base amplitude of the pressure force
force_modulation_amplitude = 0.5  # e.g., 0.5 means 50% variation
force_variation_frequency = 1.0  # rad/s for the sinusoidal variation of F0

k = 1 # Damping coefficient (formerly c_damping, now k)
g = 9.81
# If = 0.9  # Inertia of the flywheel -> This is now I_flywheel
# c_damping = 0.7  # Damping coefficient for the flywheel load -> This is now k

F0_MODULATION_MODE = "current_sin"  # Options: "current_sin", "sin", "absolute_sin", "exp_positive", "exp_negative", "linear_positive", "linear_negative", "constant"

# Function to calculate the time-varying modulation factor for F0
def calculate_force_modulation_factor(time, mode, amplitude, frequency):
    """ 
    Calculates the modulation factor for F0 based on the selected mode.
    This factor is typically force_modulation_amplitude * some_function(frequency * time).
    The final F0 magnitude used in Q will be F0_amplitude * (1 + factor).
    """
    scaled_time = frequency * time

    match mode:
        case "current_sin": # Your existing sinusoidal variation
            factor = amplitude * np.sin(scaled_time)
        case "sin": # Standard sine wave modulation
            factor = amplitude * np.sin(scaled_time)
        case "absolute_sin": # Absolute sine wave modulation (always positive modulation)
            factor = amplitude * np.abs(np.sin(scaled_time))
        case "exp_positive": # Exponentially increasing modulation
            factor = amplitude * (np.exp(scaled_time/frequency * 0.1) -1) # Scaled to start near 0
        case "exp_negative": # Exponentially decaying modulation
            factor = amplitude * (np.exp(-scaled_time/frequency * 0.1) -1) # Scaled to start near 0 and go negative
        case "linear_positive": # Linearly increasing modulation
            factor = amplitude * (scaled_time/frequency * 0.1) # Scaled rate
        case "linear_negative": # Linearly decreasing modulation
            factor = amplitude * (-scaled_time/frequency * 0.1) # Scaled rate
        case "constant": # No time variation in modulation, just a constant offset factor
            factor = amplitude 
        case _:
            print(f"Warning: Invalid F0_MODULATION_MODE '{mode}'. Defaulting to 'current_sin'.")
            factor = amplitude * np.sin(scaled_time) # Default to original behavior
    return factor

# %%
t = sp.symbols('t')

# Define time-varying F0
# F0_t = F0_amplitude * (1 + force_modulation_amplitude * sp.sin(force_variation_frequency * t)) # This will be replaced
current_F0_base_magnitude = sp.symbols('current_F0_base_magnitude') # New symbolic placeholder

x1, x2, y1, y2, theta1, theta2, x3, y3 = dynamicsymbols('x1 x2 y1 y2 theta1 theta2 x3 y3')
q = sp.Matrix([x1, y1, theta1, x2, y2, theta2, x3, y3])
dq = q.diff(t)

x_com_1 = sp.Matrix([x1, y1])
x_com_2 = sp.Matrix([x2, y2])
x_com_3 = sp.Matrix([x3, y3])

R = lambda theta: sp.Matrix([[sp.cos(theta), -sp.sin(theta)], [sp.sin(theta), sp.cos(theta)]])

# Update mass matrix to include flywheel inertia
M = np.diag([m1, m1, I1 + I_flywheel, m2, m2, I2, m3, m3])  # Added I_flywheel to I1
W = np.linalg.inv(M)
# Update Q to use the new current_F0_base_magnitude placeholder
Q = sp.Matrix([0, -m1*g, -k*theta1.diff(t), 0, -m2*g, 0, 0, -m3*g + current_F0_base_magnitude * sp.cos(theta1)])

# %%
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

# %%
J = C.jacobian(q)     
dq = q.diff(t)        
dC = J @ dq
dJ = dC.jacobian(q)
JWJT = J @ W @ J.T
RHS = -dJ @ dq - J @ W @ Q - 1 * C - 1 * dC

JWJT_fn = sp.lambdify(args=(q, dq), expr=JWJT)
RHS_fn = sp.lambdify(args=(q, dq, current_F0_base_magnitude), expr=RHS)
C_fn = sp.lambdify(args=(q, dq), expr=C)    
J_fn = sp.lambdify(args=(q, dq), expr=J)   
dC_fn = sp.lambdify(args=(q, dq), expr=dC)  
dJ_fn = sp.lambdify(args=(q, dq), expr=dJ)
Q_fn = sp.lambdify(args=(q, dq, current_F0_base_magnitude), expr=Q)

# %%
dtheta1 = 0.5
initial_position_body_1 = np.array([0, L1/2, np.pi/2])
initial_position_body_2 = np.array([0, L1 + L2/2, np.pi/2])
initial_position_body_3 = np.array([0, L1 + L2])
initial_velocity_body_1 = np.array([0, 0, dtheta1]) # To start the engine
initial_velocity_body_2 = np.array([0, 0, 0])
initial_velocity_body_3 = np.array([0, 0])
x0 = np.concatenate((initial_position_body_1, initial_position_body_2, initial_position_body_3,
                    initial_velocity_body_1, initial_velocity_body_2, initial_velocity_body_3))

# %% [markdown]
# Calculate initial conditions for the system

# %%
import scipy.optimize as opt

x, _ = np.split(x0, 2)
def optimiser(b):
    dx1, dy1, dx2, dy2, dtheta2, dx3, dy3 = b
    dq = np.array([dx1, dy1, dtheta1, dx2, dy2, dtheta2, dx3, dy3])
    val = dC_fn(x, dq).flatten()
    return val

initial_guess = np.array([0, 0, 0, 0, 0, 0, 0])
result = opt.root(optimiser, initial_guess)
print(result)

b = result.x
dx = np.array([b[0], b[1], dtheta1, b[2], b[3], b[4], b[5], b[6]])

C_val = C_fn(x, dx)
dC_val = dC_fn(x, dx)

print(f'Position constraint: {C_val}')
print(f'Velocity constraint: {dC_val}')
assert np.allclose(C_val, 0), "Initial position constraint violated"
assert np.allclose(dC_val, 0), "Initial velocity constraint violated"
x0 = np.concatenate((x, dx))
x0

# %%
def piston_engine(t, state):
    '''
    This function returns the derivative of the state vector for the system

    Parameters:
    t: float
        The current time
    state: numpy array
        The current state of the system
        The vector is arranged as [q, dq]
        where q is the position vector and dq is the derivative of the position vector
    '''

    q, dq = np.split(state, 2)

    # Calculate the current F0 base magnitude based on time and mode
    modulation_factor = calculate_force_modulation_factor(time=t, 
                                                          mode=F0_MODULATION_MODE, 
                                                          amplitude=force_modulation_amplitude, 
                                                          frequency=force_variation_frequency)
    actual_F0_base_magnitude = F0_amplitude * (1 + modulation_factor)

    # Solve for lambda 
    lam = np.linalg.solve(JWJT_fn(q,dq), RHS_fn(q, dq, actual_F0_base_magnitude))

    # Solve for constraint forces 
    Qhat = J_fn(q, dq).T @ lam

    # Calculate accelerations
    ddq = W @ (Q_fn(q, dq, actual_F0_base_magnitude) + Qhat)
    ddq = ddq.flatten()

    return np.concatenate((dq, ddq))

# Test run
piston_engine(0, x0)

# %%
t_span = (0, 30)
t_eval = np.linspace(*t_span, 500)
sol = solve_ivp(piston_engine, t_span, x0, atol=1e-7, rtol=1e-7, method='BDF', t_eval=t_eval)

# %% [markdown]
# Animation

# %%
# Class for drawing the box
class Box:
    def __init__(self, width, height, color='b'):
        self.width = width
        self.height = height
        self.color = color
        self.offset = -np.array([width/2, height/2])

    def first_draw(self, ax):
        corner = np.array([0, 0])
        self.patch = plt.Rectangle(corner, 0, 0, angle=0, 
                        rotation_point='center', color=self.color, animated=True)
        ax.add_patch(self.patch)
        self.ax = ax
        return self.patch
    
    def set_data(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def update(self, i):
        x, y, theta = self.x[i], self.y[i], self.theta[i]
        theta = np.rad2deg(theta)

        # The rectangle is drawn from the left bottom corner
        # So, we need to calculate the corner position
        corner = np.array([x, y]) + self.offset

        # Update the values for the rectangle
        self.patch.set_width(self.width)
        self.patch.set_height(self.height)
        self.patch.set_xy(corner)
        self.patch.set_angle(theta)
        return self.patch

# Class for drawing a circle (for the flywheel)
class CircleBody:
    def __init__(self, radius, color='purple'):
        self.radius = radius
        self.color = color

    def first_draw(self, ax):
        # A circle is defined by its center and radius
        self.patch = plt.Circle((0, 0), self.radius, color=self.color, animated=True, zorder=-1) # zorder to draw it behind box1
        ax.add_patch(self.patch)
        self.ax = ax
        return self.patch
    
    def set_data(self, x, y, theta): # x, y are center coordinates
        self.x_center = x
        self.y_center = y
        # theta is kept for consistency with Box, though not used for a simple circle's appearance directly
        self.theta = theta 

    def update(self, i):
        center_x, center_y = self.x_center[i], self.y_center[i]
        self.patch.center = (center_x, center_y)
        # A simple circle doesn't visually rotate, but it moves with body 1
        return self.patch

# %%
### NOTE: This might take a while to run.

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility
from matplotlib.animation import FuncAnimation
# from IPython.display import HTML # No longer needed for plt.show()

# Global variables for stroke counting
stroke_count = 0
previous_dy3_sign = 0

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.25) # Adjust layout to make space for parameters
# Close the figure as we will be making an animation
plt.close()

# Set bounding limits 
ax.set_ylim(-0.6, 2.1)
ax.set_xlim(-0.6, 0.6)
ax.set_aspect('equal')

# Get the position and angle of the two bodies
x1_sol, y1_sol, theta1_sol = sol.y[:3] # Renamed to avoid conflict
x2_sol, y2_sol, theta2_sol = sol.y[3:6] # Renamed to avoid conflict
x3_sol, y3_sol = sol.y[6:8]             # Renamed to avoid conflict
theta3_sol = np.zeros_like(x3_sol)      # Renamed to avoid conflict

box1 = Box(L1, 0.05, 'cyan') # Made thicker and cyan for diagnostics
box2 = Box(L2, 0.01, 'r')
box3 = Box(0.1, 0.3, 'g')
flywheel_viz = CircleBody(radius=L1, color='magenta') # Radius is now L1

# Prepare static data for the flywheel centered at (0,0)
num_frames = len(sol.t)
flywheel_x_center = np.zeros(num_frames)
flywheel_y_center = np.zeros(num_frames)
flywheel_theta = np.zeros(num_frames) # Angle doesn't matter for circle, but pass for consistency

box1.set_data(x1_sol, y1_sol, theta1_sol) 
box2.set_data(x2_sol, y2_sol, theta2_sol) 
box3.set_data(x3_sol, y3_sol, theta3_sol) 
flywheel_viz.set_data(flywheel_x_center, flywheel_y_center, flywheel_theta) # Flywheel is centered at (0,0)

boxes = [flywheel_viz, box1, box2, box3]

# Parameter string for display
param_string = (
    f"Parameters:\n"
    f"m1={m1:.2f}, m2={m2:.2f}, m3={m3:.2f}\n"
    f"L1={L1:.2f}, L2={L2:.2f}\n"
    f"I1={I1:.2f}, I2={I2:.2f}\n"
    f"m_fly={m_flywheel:.2f}, I_fly={I_flywheel:.2f}\n"
    f"F0_amp={F0_amplitude:.2f}, F_mod_amp={force_modulation_amplitude:.2f}\n"
    f"F_var_freq={force_variation_frequency:.2f}, F_mode='{F0_MODULATION_MODE}'\n"
    f"k_damp={k:.2f}, g={g:.2f}"
)

def init():
    ax.clear() # Clear entire axes for a fresh start
    ax.set_ylim(-0.6, 2.1) # Re-apply limits after clearing
    ax.set_xlim(-0.6, 0.6)
    ax.set_aspect('equal')
    ax.set_title("t=0.00s | Rot: 0.0 | Strokes: 0", fontsize=12) # Simplified title

    # Add parameter text to the figure
    fig.text(0.02, 0.02, param_string, fontsize=8, va='bottom', ha='left', family='monospace')

    patches_to_return = []
    for B in boxes:
        patch = B.first_draw(ax)
        patches_to_return.append(patch)
    return patches_to_return

def animate(i):
    global stroke_count, previous_dy3_sign

    current_theta1 = theta1_sol[i]
    rotations = current_theta1 / (2 * np.pi)

    current_dy3 = sol.y[15, i]
    current_dy3_sign = np.sign(current_dy3)

    if current_dy3_sign != previous_dy3_sign and current_dy3_sign != 0:
        if previous_dy3_sign != 0:
            stroke_count += 1
        previous_dy3_sign = current_dy3_sign
    
    ax.set_title(f"t={sol.t[i]:.2f}s | Rot: {rotations:.1f} | Strokes: {stroke_count}", fontsize=12) # Simplified title

    for box in boxes:
        box.update(i)
    patches = [box.patch for box in boxes]
    return patches

# Set the interval between frames
dt = sol.t[1] - sol.t[0]

# Create the animation
anim = FuncAnimation(fig, animate, frames=len(sol.t), init_func=init, blit=True, interval=1000*dt)

# Save the animation to a file first
anim.save('piston_engine_modified.gif', writer='pillow', fps=30) # Changed filename

# Then display the animation
plt.show()