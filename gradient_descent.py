from differences import *
import copy
import cv2 as cv

# Graient descent
def gradient_descent(psi, phi, grad_phi_X, grad_phi_Y, a):
    psi_x = Dx_central(psi)
    psi_y = Dy_central(psi)
    psi_xy = Dx_central(psi_y)
    psi_xx = Dx_forward(Dx_backward(psi))
    psi_yy = Dy_forward(Dy_backward(psi))  # Partial derivativea
    # Diffusion term
    diffusion = phi * (psi_yy * psi_x ** 2 - 2 * psi_x * psi_y * psi_xy + psi_xx * psi_y ** 2)/(psi_x ** 2 + psi_y ** 2)
    psi_temp = copy.copy(psi)
    psi_temp[psi < 0.5] = 1
    psi_temp[psi > -0.5] = 0
    diffusion_mat = psi_temp * diffusion / (np.sqrt(psi_x ** 2 + psi_y ** 2))
    diffusion_energy = np.sum(diffusion_mat)

    # Advection term
    advection = upwind_difference(grad_phi_X, grad_phi_Y, psi)
    advection_mat = psi_temp * advection / (np.sqrt(psi_x ** 2 + psi_y ** 2))
    advection_energy = np.sum(advection_mat)

    # Erosion term
    erosion = a * entropy_upwind(phi, psi)
    erosion_mat = psi_temp * erosion / (np.sqrt(psi_x ** 2 + psi_y ** 2))
    erosion_energy = np.sum(erosion_mat)
    delta_t = CFL(phi, grad_phi_X, grad_phi_Y, a)
    grad_energy = advection + diffusion + erosion

    # Update psi
    psi_new = psi + delta_t * grad_energy
    return psi_new, diffusion_energy, advection_energy, erosion_energy

# Computing energy
def compute_energy(psi, phi, a):
    psi_temp_1 = copy.copy(psi)
    psi_temp_2 = copy.copy(psi)
    psi_temp_1[psi < 0.5] = 1
    psi_temp_1[psi >= 0.5] = 0
    psi_temp_2[psi > -0.5] = 1
    psi_temp_2[psi <= -0.5] = 0
    psi_temp = psi_temp_1 * psi_temp_2
    psi_temp_in = copy.copy(psi)
    psi_temp_in[psi < 0] = 1
    psi_temp_in[psi >= 0] = 0
    energy_mat = psi_temp * phi + a * psi_temp_in * phi
    energy = np.sum(energy_mat)
    return energy
