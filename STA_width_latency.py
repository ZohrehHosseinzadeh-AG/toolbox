# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 12:16:44 2021

@author: Admin
"""
from scipy.signal import chirp, find_peaks, peak_widths
import matplotlib.pyplot as plt

x=Mean_sta[c][:101].flatten()

peaks, _ = find_peaks(x,height=max(x))
npeaks, _ = find_peaks(-x,height=(max(-x)-10))

results_half = peak_widths(x, peaks, rel_height=0.5)
results_half[0]  # widths

results_full = peak_widths(x, peaks, rel_height=.5)
results_full_n = peak_widths(-x, npeaks, rel_height=.5)
results_full_n= list(results_full_n) 
results_full_n[1]= -results_full_n[1]

plt.plot(x)
plt.plot(peaks, x[peaks], "x")
plt.plot(npeaks, x[npeaks], "x")

#plt.hlines(*results_half[1:], color="C2")
plt.hlines(*results_full[1:], color="C3")
plt.hlines(*results_full_n[1:], color="C2")

plt.show()