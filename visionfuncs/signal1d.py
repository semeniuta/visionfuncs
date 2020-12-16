import numpy as np
from scipy import signal
from scipy import interpolate


def find_good_peaks(s, base_diff_threshold=20):
    """
    Find good peaks of a 1D signal based on 
    discarding those with small absolute difference 
    between left and right bases
    (obtained from scipy.signal.peak_prominences).
    """
    
    peaks, _ = signal.find_peaks(s)
    
    prominences, bases_left, bases_right = signal.peak_prominences(s, peaks)
    
    base_diff_abs = np.abs(bases_left - bases_right)
    
    return peaks[base_diff_abs > base_diff_threshold]


def downsample_by_interpolate(s, n):
    """
    Dowsample a 1D signal s to a signal 
    of size n by performing linear
    interpolation of the original signal. 
    """

    sig_len = len(s)
    domain = np.arange(sig_len)
    f = interpolate.interp1d(domain, s)

    step = sig_len / n
    x = np.arange(0, sig_len, step)
    
    return f(x)
