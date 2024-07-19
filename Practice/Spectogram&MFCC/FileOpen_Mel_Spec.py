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

mfcc1 = librosa.feature.mfcc(y=y1, sr=sr, n_mfcc=40, hop_length = 128)
#plt.figure(figsize=(16,6))
plt.subplot(2,1,1)

librosa.display.specshow(mfcc1,sr=sr, hop_length=128)
plt.colorbar()


y2 , sr = librosa.load('D:/A_바탕화면2/24년1학기강의/기계학습/과제4/a3.wav', sr=None)

mfcc2 = librosa.feature.mfcc(y=y2, sr=sr, n_mfcc=40, hop_length = 128)
#plt.figure(figsize=(16,6))
plt.subplot(2,1,2)

librosa.display.specshow(mfcc2,sr=sr, hop_length=128)
plt.colorbar()
plt.show()
