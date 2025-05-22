#Original Code


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
F0 = 50
k = 1
g = 9.81

# %%
t = sp.symbols('t')
x1, x2, y1, y2, theta1, theta2, x3, y3 = dynamicsymbols('x1 x2 y1 y2 theta1 theta2 x3 y3')
q = sp.Matrix([x1, y1, theta1, x2, y2, theta2, x3, y3])
dq = q.diff(t)

x_com_1 = sp.Matrix([x1, y1])
x_com_2 = sp.Matrix([x2, y2])
x_com_3 = sp.Matrix([x3, y3])

R = lambda theta: sp.Matrix([[sp.cos(theta), -sp.sin(theta)], [sp.sin(theta), sp.cos(theta)]])

M = np.diag([m1, m1, I1, m2, m2, I2, m3, m3])
W = np.linalg.inv(M)
Q = sp.Matrix([0, -m1*g, -k*theta1.diff(t), 0, -m2*g, 0, 0, -m3*g + F0 * sp.cos(theta1)])

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
RHS_fn = sp.lambdify(args=(q, dq), expr=RHS)
C_fn = sp.lambdify(args=(q, dq), expr=C)    
J_fn = sp.lambdify(args=(q, dq), expr=J)   
dC_fn = sp.lambdify(args=(q, dq), expr=dC)  
dJ_fn = sp.lambdify(args=(q, dq), expr=dJ)
Q_fn = sp.lambdify(args=(q, dq), expr=Q)

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

    # Solve for lambda 
    lam = np.linalg.solve(JWJT_fn(q,dq), RHS_fn(q,dq))

    # Solve for constraint forces 
    Qhat = J_fn(q, dq).T @ lam

    # Calculate accelerations
    ddq = W @ (Q_fn(q, dq) + Qhat)
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

# %%
### NOTE: This might take a while to run.

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Global variables for stroke counting
stroke_count_org = 0
previous_dy3_sign_org = 0

fig, ax = plt.subplots()
# Close the figure as we will be making an animation
plt.close()

# Set bounding limits 
ax.set_ylim(-0.6, 2.1)
ax.set_xlim(-0.6, 0.6)
ax.set_aspect('equal')

# Get the position and angle of the two bodies
x1, y1, theta1 = sol.y[:3]
x2, y2, theta2 = sol.y[3:6]
x3, y3 = sol.y[6:8]
theta3 = np.zeros_like(x3)

box1 = Box(L1, 0.01, 'b')
box2 = Box(L2, 0.01, 'r')
box3 = Box(0.1, 0.3, 'g')

box1.set_data(x1, y1, theta1)
box2.set_data(x2, y2, theta2)
box3.set_data(x3, y3, theta3)

boxes = [box1, box3, box2]

def init():
    ax.set_title("t=0.00 sec | Rot: 0.0 | Strokes: 0", fontsize=15)
    for box in boxes:
        box.first_draw(ax)
    patches = [box.patch for box in boxes]
    return patches

def animate(i):
    ''' Draw the i-th frame of the animation'''
    global stroke_count_org, previous_dy3_sign_org

    current_theta1 = sol.y[2, i]
    rotations = current_theta1 / (2 * np.pi)

    # Piston stroke counting for SystemBOrg
    # y3 is q[7] (index 7), so dy3 is sol.y[7+8=15, i]
    current_dy3_org = sol.y[15, i] 
    current_dy3_sign_org = np.sign(current_dy3_org)

    if current_dy3_sign_org != previous_dy3_sign_org and current_dy3_sign_org != 0:
        if previous_dy3_sign_org != 0: # Only count if it was previously moving
            stroke_count_org += 1
        previous_dy3_sign_org = current_dy3_sign_org

    ax.set_title(f"t={sol.t[i]:.2f} sec | Rot: {rotations:.1f} | Strokes: {stroke_count_org}", fontsize=15)

    for box in boxes:
        box.update(i)
    patches = [box.patch for box in boxes]
    return patches

# Set the interval between frames
dt = sol.t[1] - sol.t[0]

# Create the animation
anim = FuncAnimation(fig, animate, frames=len(sol.t), init_func=init, blit=True, interval=1000*dt)

# Save the animation to a file first
anim.save('piston_engine_org.gif', writer='pillow', fps=30)

# Then display the animation
plt.show()


