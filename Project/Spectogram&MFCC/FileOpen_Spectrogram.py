from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np
import math
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt

import librosa
import librosa.display


y1 , sr = librosa.load('D:/A_바탕화면2/24년1학기강의/기계학습/과제4/a2.wav', sr=None)


D1 = np.abs(librosa.stft(y1, n_fft=512, hop_length=128))

print(D1.shape)

DB1 = librosa.amplitude_to_db(D1, ref=np.max) #amplitude(진폭) -> DB(데시벨)로 바꿔라
plt.subplot(2,1,1)
#plt.figure(figsize=(16,6))
librosa.display.specshow(DB1,sr=sr, hop_length=128, x_axis='time', y_axis='log')
plt.colorbar()



y2 , sr = librosa.load('D:/A_바탕화면2/24년1학기강의/기계학습/과제4/a3.wav', sr=None)
D2 = np.abs(librosa.stft(y2, n_fft=512, hop_length=128))

print(D2.shape)

DB2 = librosa.amplitude_to_db(D2, ref=np.max) #amplitude(진폭) -> DB(데시벨)로 바꿔라
plt.subplot(2,1,2)
#plt.figure(figsize=(16,6))
librosa.display.specshow(DB2,sr=sr, hop_length=128, x_axis='time', y_axis='log')
plt.colorbar()
plt.show()


