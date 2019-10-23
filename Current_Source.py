import numpy as np


def get_k_beta(kz, kt, v):
    k_beta = kz - (kt/v)
    return k_beta


def get_k_eta(kz, kt, v):
    k_eta = kz + (kt/v)
    return k_eta


#def derv_potential()


def fourier_potential_derv(volt_derv):
    fourier_voltage_dev = np.fft.fft(volt_derv)
    return fourier_voltage_dev


def compute_source(kz, kt, v, L1, L2, z_i):
    keta = get_k_eta(kz, kt, v)
    kbeta = get_k_beta(kz, kt, v)

    current = complex(0, 1)*kz*np.exp(-complex(0, 1)*kz*z_i)*fourier_potential_derv(kt, v, voltage_derv)\
              *(np.exp(-complex(0, 1)*keta*L1/2)*(np.sin(keta*L1/2))/(keta/2)
                - np.exp(complex(0, 1)*kbeta*L2/2)*(np.sin(kbeta*L2/2))/(kbeta/2))
    return current


