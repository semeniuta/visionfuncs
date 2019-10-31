import numpy as np
from scipy import signal


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