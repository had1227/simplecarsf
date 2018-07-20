import numpy as np

def spmax(logits):

    z_sorted = np.sort(logits)[::-1]
    z_cumsum = np.cumsum(z_sorted)
    k = np.arange(1,len(z_sorted)+1)

    z_check = 1 + k * z_sorted > z_cumsum
    k_z = np.sum(z_check)

    tau_sum = np.sum(z_sorted * z_check)
    tau = tau_sum / k_z

    z_spmax = 0.5 + 0.5 * np.sum(z_sorted[:k_z]**2-tau**2)
    
    return z_spmax
