import numpy as np
from scipy import signal

def apply_low_pass(data, cutoff, fs, order=5):
    """
    Apply a low-pass Butterworth filter to the data.
    
    :param data: Input signal (numpy array)
    :param cutoff: Cutoff frequency in Hz
    :param fs: Sampling frequency in Hz
    :param order: Order of the filter
    :return: Filtered signal
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.lfilter(b, a, data)
    return y

def apply_notch_filter(data, notch_freq, fs, quality_factor=30):
    """
    Apply a notch filter to remove specific noise (e.g., 50Hz/60Hz mains hum).
    
    :param data: Input signal
    :param notch_freq: Frequency to remove
    :param fs: Sampling frequency
    :param quality_factor: Quality factor (higher = narrower notch)
    :return: Filtered signal
    """
    nyq = 0.5 * fs
    freq = notch_freq / nyq
    b, a = signal.iirnotch(freq, quality_factor)
    y = signal.lfilter(b, a, data)
    return y

def moving_average(data, window_size=5):
    """
    Simple moving average filter for smoothing.
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
