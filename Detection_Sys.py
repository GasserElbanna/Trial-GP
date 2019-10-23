import numpy as np
from scipy.special import jv


def kz_transform(kz, kth, alpha, R):
    k = ((kth*np.sin(alpha))/R) + (kz*np.cos(alpha))
    return k


def kth_transform(kz, kth, alpha, R):
    k = (kth*np.cos(alpha)) - (kz*np.sin(alpha)*R)
    return k


def dist_theta(dist, R):
    distance = dist / R
    return distance


def rect_elec(length, width, kz, kth, R):
    trans_fn = np.sinc((kth*length)/(2*np.pi*R))*np.sinc((kz*width)/(2*np.pi))
    return trans_fn


def elliptical_elec(a, b, kz, kth, R):
    trans_fn = 2*((2*jv(1, (np.sqrt(((a*kth/R)**2)+((b*kz)**2))))))/(np.sqrt(((a*kth/R)**2)+((b*kz)**2)))
    return trans_fn


def elec_trans_fn(kz, kth, weight, i, u, size_tf, dz, dth):
    trans_fn = weight[i+1, u+1] * np.exp(-complex(0, 1)*kz*i*dz)*np.exp(-complex(0, 1)*kth*u*dth)
    return trans_fn


def compute_detection_sys(kz, kth, weighting, z_i, th_u, alpha, dz, d, Relec, type, a, b):
    dth = dist_theta(d, Relec)

    # if type == 0:
    #     size_tf = rect_elec(a, b, k_z, k_th, Relec)
    # else:
    #     size_tf = elliptical_elec(a, b, k_z, k_th, Relec)
    trans_fn = 0
    alpha = -5
    for u in range(th_u):
        for i in range(z_i):
            k_z = kz_transform(kz, kth, alpha, Relec)
            k_th = kth_transform(kz, kth, alpha, Relec)
            trans_fn += elec_trans_fn(k_z, k_th, weighting, i-1, u-1, 0, dz, dth)
        alpha += 5

    return trans_fn