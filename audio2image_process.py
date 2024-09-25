import numpy as np
import wave
import os
from matplotlib.image import imread
from brian2 import *
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
# This file draws a large number of spectrograms of audio files and saves them as 28×28 pixel images
COOKED_DIR = 'E:/recordings/'# The file where the "free spoken digit dataset" is saved, you can easily download it from Internet
COOKED_DIR2 = 'E:/photos/'# The file where you save the image
i = 1
for root, dirs, files in os.walk(COOKED_DIR):
    print("Root =", root, "dirs =", dirs, "files =", files)
    for filename in files:
        path_one = COOKED_DIR + filename
        f = wave.open(path_one, 'rb')
        params = f.getparams()  # Returns all audio parameters at once, including number of channels, sample width, frame rate, and number of frames
        nchannels, sampwidth, framerate, nframes = params[:4]  # Number of channels, sample width, frame rate, and number of frames
        str_data = f.readframes(nframes)  # Reads the specified length (in sample points) and returns a string of audio data
        waveData = np.frombuffer(str_data, dtype=np.int16)  # Converts the string to int
        waveData = waveData * 1.0 / (max(abs(waveData)))  # Normalize the amplitude of the wave
        plt.rcParams['figure.figsize'] = (6.2, 6.2)  # Set the figure size to 6.2, 6.2
        plt.rcParams['savefig.dpi'] = 50  # Set the image resolution
        plt.specgram(waveData, cmap='gray', NFFT=512, Fs=framerate, noverlap=500, scale_by_freq=True, sides='default')
        plt.axis('off')
        name = str(i)  # Create a name
        plt.savefig(COOKED_DIR2 + "photo28" + name + ".jpg", bbox_inches='tight', pad_inches=-0.1, dpi=6.25)  # Save the image with removed white borders
        i += 1
        print(i)
