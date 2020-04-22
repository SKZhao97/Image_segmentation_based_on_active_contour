import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from differences import *
# np.set_printoptions(threshold=np.inf)

# Initialization
def init_levelset(size):
    x = np.linspace(1, size[1], size[1])
    y = np.linspace(1, size[0], size[0])
    X, Y = np.meshgrid(x, y)
    # Plot a circle
    r = min(size[0], size[1])/2
    radius = r
    center = (size[0]/2 - 25, size[1]/2 + 5)
    psi0 = np.sqrt((X - [center[1]]) ** 2 + (Y - [center[0]]) ** 2) - radius * 0.875
    return psi0

# Reinitialization
def reinitialization(psi, it):
    e = 1
    sign_function = psi / np.sqrt(psi ** 2 + e ** 2)  # Sign distance function
    psi_new = psi

    # Update psi
    for i in range(it):
        psi_delta = entropy_upwind(sign_function, psi_new, True)  # Delta t
        delta_t = 1/np.sqrt(2)
        psi_new = psi_new + delta_t * (sign_function - psi_delta)  # iterate
    return psi_new

# Plot image with psi
def image_psi(img, psi):
    plt.imshow(img, cmap='Greys_r')
    plt.contour(psi, 0, colors='red')
    plt.show()


