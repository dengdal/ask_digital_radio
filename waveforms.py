# -*- coding: utf-8 -*-

import numpy as np

SPWL = 100

def gen_wavelength(samples = SPWL, angle = 0):
    a = np.linspace(0,2*np.pi, samples)
    a = np.roll(a, samples * (angle / 2*np.pi))
    return np.sin(a)

def shift_phase(waveform, angle = 0):
    samples = np.shape(waveform)[0]
    return np.roll(waveform, -(int(samples * (angle / (2*np.pi)))))

def modulate_qpsk_symbol(waveform, data):
    return shift_phase(waveform, data/4 * 2*np.pi)

def modulate_data_to_qpsk_symbols(data, samples_per_wavelength = SPWL):   
    modulated_waveform = np.zeros(samples_per_wavelength * int(np.size(data)/2)) # two bits per QPSK symbol
    wave = gen_wavelength(samples_per_wavelength) # basic unmodulated wave
    for index, value in enumerate(data):
       if index % 2 == 0:
           data_to_modulate = value * 2 # MSB first
       else:
           data_to_modulate += value # LSB last
           new_wave = modulate_qpsk_symbol(wave, data_to_modulate)
           # add new wave to total waveform
           modulated_waveform[int(((index-1)/2*samples_per_wavelength)): int(((index-1)/2*samples_per_wavelength + samples_per_wavelength))] = new_wave
    return modulated_waveform

def get_one_wave(modulated_waveform, index, samples_per_wavelength = SPWL):
    return modulated_waveform[index*samples_per_wavelength: (index+1)*samples_per_wavelength]

def demodulate_qpsk_symbol(wave, samples_per_wavelength = SPWL):
    if np.abs(wave[0]) < 0.5:
        if wave[int(samples_per_wavelength/4)] > 0.5:
            data = [0,0]
        else:
            data = [1,0]
    else:
        if wave[0] > 0:
            data = [0,1]
        else:
            data = [1,1]
    return data

def demodulate_qpsk_waveform(waveform, samples_per_wavelength = SPWL):
    data = []
    wavelengths = int(np.shape(waveform)[0] / samples_per_wavelength)
    for index in range(wavelengths):
        wave = get_one_wave(waveform, index, samples_per_wavelength)
        data += demodulate_qpsk_symbol(wave, samples_per_wavelength)
    return data

def noisy_channel(waveform, snr):
    noise = (1/snr) * np.random.randn(np.shape(waveform)[0])
    return (waveform + noise)

def matrixmult (A, B):
    C = [[0 for row in range(len(A))] for col in range(len(B[0]))]
    for i in range(len(A)): # rows in A
        for j in range(len(B[0])): # cols in B
            for k in range(len(B)): # rows in B
                C[i][j] += A[i][k]*B[k][j]
    return C
            