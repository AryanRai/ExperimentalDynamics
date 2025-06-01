import numpy as np
import matplotlib.pyplot as plt

def tests(F):
    l = 1
    r = 0.5

    x = np.linspace(0, 2 * np.pi, 360)  # 1 full revolution in radians
    y = np.zeros(360)

    for i in range(360):
        angle = x[i]
        y[i] = F * np.cos(angle) * r * np.sin(angle) * (np.sqrt(l**2 - (r**2) * (np.sin(angle)**2)) / l)

    return x, y

# Forces
F1 = 100
F2 = 200
F3 = 400

# Get data
X1, Y1 = tests(F1)
X2, Y2 = tests(F2)
X3, Y3 = tests(F3)

# Plotting
plt.figure()
plt.plot(X1, Y1, 'k-', label='F = 100N')
plt.plot(X2, Y2, 'r-', label='F = 200N')
plt.plot(X3, Y3, 'b-', label='F = 400N')
plt.xlabel('Crank Angle (rad)')
plt.ylabel('Output Torque (Nm)')
plt.title('Output Torque During a Revolution at Different Force Values')
plt.legend()
plt.xlim([0, 6.3])
plt.grid(True)
plt.show()