import numpy as np
import copy

# Central difference of x
def Dx_central(m):
    y0, x0 = m.shape
    f = np.pad(m, ((0, 0), (1, 1)), 'edge')
    y, x = f.shape
    fx = np.zeros([y0, x0])
    fx[:, 0:x0] = 0.5 * (f[:, 2:x] - f[:, 0:x-2])
    return fx

# Central difference of y
def Dy_central(my):
    y0, x0 = my.shape
    f = np.pad(my, ((1, 1), (0, 0)), 'edge')
    y, x = f.shape
    fy = np.zeros([y0, x0])
    fy[0:y0, :] = 0.5 * (f[2:y, :] - f[0:y-2, :])
    return fy


# Forward difference of x
def Dx_forward(m):
    y0, x0 = m.shape
    f = np.pad(m, ((0, 0), (0, 1)), 'edge')
    y, x = f.shape
    fx = np.zeros([y0, x0])
    fx[:, 0:x0] = f[:, 1:x] - f[:, 0:x-1]
    return fx

# Backward difference of x
def Dx_backward(m):
    y0, x0 = m.shape
    f = np.pad(m, ((0, 0), (1, 0)), 'edge')
    y, x = f.shape
    fx = np.zeros([y0, x0])
    fx[:, 0:x0] = f[:, 1:x] - f[:, 0:x-1]
    return fx

# Forward difference of y
def Dy_forward(m):
    y0, x0 = m.shape
    f = np.pad(m, ((0, 1), (0, 0)), 'edge')
    y, x = f.shape
    fy = np.zeros([y0, x0])
    fy[0:y0, :] = f[1:y, :] - f[0:y-1, :]
    return fy

# Backward difference of y
def Dy_backward(m):
    y0, x0 = m.shape
    f = np.pad(m, ((1, 0), (0, 0)), 'edge')
    y, x = f.shape
    fy = np.zeros([y0, x0])
    fy[0:y0, :] = f[1:y, :] - f[0:y-1, :]
    return fy

# Upwind difference
def upwind_difference(v1, v2, m):
    v1_temp_1 = copy.copy(v1)
    v1_temp_2 = copy.copy(v1)
    v2_temp_1 = copy.copy(v2)
    v2_temp_2 = copy.copy(v2)
    v1_temp_1[v1 < 0] = 0
    v1_temp_2[v1 > 0] = 0
    v2_temp_1[v2 < 0] = 0
    v2_temp_2[v2 > 0] = 0
    x = v1_temp_1 * Dx_forward(m) + v1_temp_2 * Dx_backward(m)
    y = v2_temp_1 * Dy_forward(m) + v2_temp_2 * Dy_backward(m)
    r = x + y
    return r

# entropy upwind differnce
def entropy_upwind(phi, m, p=None):
    Dx_f = Dx_forward(m)
    Dx_b = Dx_backward(m)
    Dy_f = Dy_forward(m)
    Dy_b = Dy_backward(m)
    Dx_f_temp = copy.copy(Dx_f)
    Dx_b_temp = copy.copy(Dx_b)
    Dy_f_temp = copy.copy(Dy_f)
    Dy_b_temp = copy.copy(Dy_b)
    Dx_f_temp_1 = copy.copy(Dx_f)
    Dx_b_temp_1 = copy.copy(Dx_b)
    Dy_f_temp_1 = copy.copy(Dy_f)
    Dy_b_temp_1 = copy.copy(Dy_b)
    Dx_f_temp[Dx_f < 0] = 0
    Dx_b_temp[Dx_b > 0] = 0
    Dy_f_temp[Dy_f < 0] = 0
    Dy_b_temp[Dy_b > 0] = 0
    Dx_f_temp_1[Dx_f > 0] = 0
    Dx_b_temp_1[Dx_b < 0] = 0
    Dy_f_temp_1[Dy_f > 0] = 0
    Dy_b_temp_1[Dy_b < 0] = 0
    phi_temp = copy.copy(phi)
    phi_temp_1 = copy.copy(phi)
    phi_temp[phi < 0] = 0
    phi_temp_1[phi > 0] = 0
    output = phi_temp * (np.sqrt(Dx_f_temp_1 ** 2 + Dx_b_temp_1 ** 2 + Dy_f_temp_1 ** 2 + Dy_b_temp_1 ** 2)) \
             + phi_temp_1 * (np.sqrt(Dx_f_temp ** 2 + Dx_b_temp ** 2 + Dy_f_temp ** 2 + Dy_b_temp ** 2))
    return output

# CFL condition calculation
def CFL(phi, grad_phi_x, grad_phi_y, a):
    delta_t_diffusion = 0.5/np.max(phi)
    delta_t_advection = 1/np.max(np.sqrt(grad_phi_x ** 2 + grad_phi_y ** 2))
    delta_t_erosion = 1/(np.sqrt(2)*a*np.max(phi))
    delta_t = min(delta_t_advection, delta_t_diffusion, delta_t_erosion)
    return delta_t
