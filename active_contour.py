import cv2 as cv
import numpy as np
from levelset import *
from gradient_descent import *

# Image
I0 = cv.imread("face.jpg")  # Image read
I1 = cv.cvtColor(I0, cv.COLOR_RGB2GRAY)  # Image transfer from RGB to Gray
size = I1.shape  # Image size
sigma = 1.2  # Scale parameter in Gauss
I = cv.GaussianBlur(I1.astype(np.float32), (7, 7), sigma)  # Gauss filter
a = 0.4  # Coefficient in the erosion term

# Gredient
Iy, Ix = np.gradient(I)  # Gradient, x and y directions
f = Ix ** 2 + Iy ** 2  # Default potential function
# f = np.sqrt(Ix ** 2 + Iy ** 2)  # Another potential function
potential = [1]/([1] + f)  # Potential function, phi = 1/1+||nabla(phi)||^2
grad_y, grad_x = np.gradient(potential)  # Gradient of phi
phi = potential
grad_phi_x = grad_x
grad_phi_y = grad_y

# Reference
reference = cv.imread("face_ref.jpg")  # The reference contour of the actual object
reference = cv.cvtColor(reference, cv.COLOR_RGB2GRAY)  # Image transfer from RGB to Gray
ret, reference = cv.threshold(reference, 175, 255, cv.THRESH_BINARY)
reference = reference/255

# Initial level set
psi = init_levelset(size)  # Init level set
image_psi(I1, psi)  # Plot the initial

# Energy
energy_array = []
diffusion_energy_gradient = []
advevtion_energy_gradient = []
erosion_energy_gradient = []
energy_gradient = []

# Error
psi_r = cv.threshold(psi, 0.5, 255, cv.THRESH_BINARY)#/[255]
psi_reverse = psi_r[1]/255
psi_bw = 1 - psi_reverse
current_error_mat = psi_bw - reference
current_error = np.sum(current_error_mat)
norm = np.sum(reference)
error = []

# Iteration
cur = 1
n = 25
flag = True
while flag:
    if cur % 10 == 0:
        psi = reinitialization(psi, 10)

    psi, diffusion_energy_grad, advection_energy_grad, erosion_energy_grad = gradient_descent(psi, phi, grad_phi_x, grad_phi_y, a)
    if cur % n == 0:
        # print(psi)
        diffusion_energy_gradient.append(diffusion_energy_grad)
        advevtion_energy_gradient.append(advection_energy_grad)
        erosion_energy_gradient.append(erosion_energy_grad)
        energy_gradient.append(diffusion_energy_grad+advection_energy_grad+erosion_energy_grad)

        energy = compute_energy(psi, phi, a)
        energy_array.append(energy)

        psi_r = cv.threshold(psi, 0.5, 255, cv.THRESH_BINARY)
        psi_reverse = psi_r[1]/255
        psi_bw = 1 - psi_reverse
        current_error_mat = psi_bw - reference
        current_error = np.sum(current_error_mat)/norm
        error.append(current_error)

        # # Stable condition
        if cur/n > 3 and abs(energy_gradient[int(cur / n) - 1] + energy_gradient[int(cur / n) - 2]
               - energy_gradient[int(cur / n) - 3] - energy_gradient[int(cur / n) - 4]) < 0.1:
            flag = False
        # iteration number condition
        # if cur == 3000:
        #     flag = False
    cur += 1

# Plot the trend
def plot_trend(mat, name):
    size = len(mat)
    x =[i for i in range(size)]
    it = [i * 25 for i in x]
    plt.figure()
    plt.plot(it, mat, 'b', label=name)
    plt.title(name + ' evolution with iteration')
    plt.xlabel('iteration')
    plt.ylabel(name)
    plt.legend()
    plt.show()

image_psi(I1, psi)
plot_trend(error, 'error')
cv.waitKey(0)