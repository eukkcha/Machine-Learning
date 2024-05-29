from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np
import math
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt

import librosa
import librosa.display

# Load the audio file
y1 , sr = librosa.load('/Users/eukkcha/Desktop/Python/MachineLearning/여자_엄마.wav', sr=None)
D1 = np.abs(librosa.stft(y1, n_fft=4096, hop_length=128)) #n_fft: 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576
print(D1.shape)

# Convert amplitude to decibels
DB1 = librosa.amplitude_to_db(D1, ref=np.max) #amplitude(진폭) -> DB(데시벨)로 바꿔라
plt.subplot(2,1,1)
#plt.figure(figsize=(16,6))
librosa.display.specshow(DB1,sr=sr, hop_length=128, x_axis='time', y_axis='log')
plt.colorbar()

# Compute the MFCCs
mfcc = librosa.feature.mfcc(y=y1, sr=sr, n_mfcc=48, hop_length = 128) #n_mfcc: 20, 40, 60, 80, 100...
#plt.figure(figsize=(16,6))
plt.subplot(2,1,2)
librosa.display.specshow(mfcc,sr=sr, hop_length=128)
plt.colorbar()

plt.show()
