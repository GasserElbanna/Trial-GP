import numpy as np
from neuron import gui, h
from scipy.special import iv, kn
from sympy import symbols, linear_eq_to_matrix
from skimage.transform import radon

from Volume_Conductor import *
from Current_Source import *
from Detection_Sys import *

#Spatial filter Input Parameters
z_elec = 3
th_elec = 3
L = 0.001
W = 0.001
z_dist = 0.005
dist = 0.005
Relec = 0.050
alpha = 5 #in degrees
elec_weighting = [[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]]
elec_weighting = np.array(elec_weighting)
##################################
#Source Input Parameters
velocity = 4 #in m/sec
fiber_length = 0.080 #in m
fiber_depth = 6 #in mm
f_sampling = 4000 #in Hz
no_fibers = 2
L1 = 0.050
L2 = 0.030
z0 = 0
##################################
#Volume Conductor Input Parameters
a = 0.020 #in m
b = 0.045 #in m
c = 0.048 #in m
d = 0.050 #in m
e = 0.045
f = 0.050
R = 0.044 #in m
volume_length = 0.125 #in m

sigsz = 1 #in S/m
sigsp = 1
sigfz = 0.05
sigfp = 0.05
sigmz = 0.55
sigmp = 0.1
sigbz = 0.02
sigbp = 0.02
####################################

a1 = get_distance(sigbz, sigbp, a)
a2 = get_distance(sigmz, sigmp, a)
b2 = get_distance(sigmz, sigmp, b)
b3 = get_distance(sigfz, sigfp, b)
c3 = get_distance(sigfz, sigfp, c)
c4 = get_distance(sigsz, sigsp, c)
d4 = get_distance(sigsz, sigsp, d)

e1 = get_distance(sigmz, sigmp, e)
e2 = get_distance(sigfz, sigfp, e)
f2 = get_distance(sigfz, sigfp, f)

Rm = get_distance(sigmz, sigmp, R)
layers = [a, b, c, d]
layers_radial_coord = [a1, a2, b2, b3, c3, c4, d4]
source_radial_coord = [R, Rm, R, R]
cond = [sigbp, sigbz, sigmp, sigmz, sigfp, sigfz, sigsp, sigsz]

# Rm = get_distance(sigmz, sigmp, R)
# layers = [e, f]
# layers_radial_coord = [e1, e2, f2]
# source_radial_coord = [Rm, R]
# cond = [sigmp, sigmz, sigfp, sigfz]

A, B, C, D, E, F, G, H, I, J, K = symbols('A B C D E F G H I J K')
sym = [A, B, C, D, E, F, G, H, I, J, K]


# Sampling and Resolutions
kz_step = np.pi/(volume_length)
kz_max = (np.pi*f_sampling)/velocity
bins = int((2*kz_max*volume_length)/np.pi)
kth_step = 1
kth_max = int((50*kth_step)/2)
z_step = 1/kz_max
th_step = int(1/kth_max)
############################


vol_cond_spatial_freq = np.zeros((30, (int(bins/2))), dtype=np.complex)
vol_cond_spatial = np.zeros((30, (int(bins/2))), dtype=np.complex)
detection_sys_spatial_freq = np.zeros((30, (int(bins/2))), dtype=np.complex)
H_glo = np.zeros((30, (int(bins/2))), dtype=np.complex)
B_spatial = np.zeros((int(bins/2)), dtype=np.complex)
z = np.linspace(0, volume_length, int(bins/2))
th = np.linspace(0, np.pi, 30)
emg = []


for w1 in range(30):
    for w2 in range(int(bins/2)):
        if w2 == 0:
            kz = 0.0000001*kz_step
            # kz = 2000
        else:
            kz = w2*kz_step
        kth = w1
        kt = velocity*kz
        # kth = 30
        vol_cond_spatial_freq[w1, w2] = compute_vol_cond(kz, kth, sym, layers, layers_radial_coord, source_radial_coord, cond)
        #vol_cond_spatial[w1, w2] = vol_cond_spatial_freq[w1, w2]*np.exp(2*np.pi*complex(0, 1)*(kz*z[w1]/128 + kth*th[w2]/30))
        detection_sys_spatial_freq[w1, w2] = compute_detection_sys(kz, kth, elec_weighting, z_elec, th_elec, alpha, z_dist, dist, Relec, 0, L, W)
        H_glo[w1, w2] = vol_cond_spatial_freq[w1, w2] * detection_sys_spatial_freq[w1, w2]
        B_spatial[w1] = radon(H_glo[w1, w2], theta=kth)
        current = compute_source(kz, kt, velocity, L1, L2, z0)
        emg = append.(np.fft.ifft2(current*B_spatial))

# for w1 in range(int(bins/2)):
#     for w2 in range(30):
#         if w1 == 0:
#             kz = 0.0000001*kz_step
#         else:
#             kz = w1*kz_step
#         kt = w2*kth_step
#         kbeta = get_k_beta(kz, kt, velocity)
#         keta = get_k_eta(kz, kt, velocity)
#         current_source[w1, w2] =
# print(vol_cond_spatial_freq)
vol_cond_spatial = np.fft.ifft2(H_glo)
# print(vol_cond_spatial)
mag = np.abs(vol_cond_spatial)
#mag = np.transpose(mag)
print(len(mag), len(mag[0]))
print(mag)
mag_max = np.amax(mag)
mag_normalized = normalize_potential(mag, mag_max)
# print(mag[0, :])
# print(len(mag), len(mag[0]))
show_impulse_response(mag_normalized, z, th)