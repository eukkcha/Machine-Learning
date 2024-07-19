import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import librosa
import seaborn as sns


# 오디오 파일을 로드하고 MFCC 특성 추출
def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=44100)  # 오디오 파일 로드
    mfcc = librosa.feature.mfcc(audio, sr=44100, n_mfcc=38)  # MFCC 특성 추출
    return mfcc.T


# 오디오 파일 경로 목록
file_paths = [
    "wav1/규보1.wav",
    "wav1/아빠1.wav",
    "wav1/엄마1.wav",
    "wav1/유상1.wav",
    "wav1/유성1.wav",
    "wav1/유찬1.wav",
    "wav1/이모1.wav",
    "wav1/지윤1.wav",
    "wav1/창희1.wav",
    "wav1/할머니1.wav",
]

models = []  # GMM 모델들을 저장할 리스트
all_features = []  # 모든 MFCC 특성들을 저장할 리스트
labels = []  # 각 파일의 라벨을 저장할 리스트

# 각 오디오 파일에 대해 GMM 모델을 학습하고 특성을 저장
for i, file_path in enumerate(file_paths):
    mfcc_features = load_audio(file_path)  # MFCC 특성 추출
    gmm = GaussianMixture(
        n_components=4, covariance_type="diag", random_state=0
    )  # GMM 모델 초기화
    gmm.fit(mfcc_features)  # GMM 모델 학습
    models.append(gmm)  # 학습된 GMM 모델 저장
    all_features.append(mfcc_features)  # 추출된 MFCC 특성 저장
    labels.extend([i] * len(mfcc_features))  # 각 파일에 대한 라벨 저장

all_features = np.vstack(all_features)  # 모든 MFCC 특성을 하나의 배열로 결합
labels = np.array(labels)  # 라벨을 배열로 변환

# PCA를 사용하여 특성 축소
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(all_features)

# PCA 결과 시각화
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=reduced_features[:, 0],
    y=reduced_features[:, 1],
    hue=labels,
    palette="tab10",
    legend="full",
)
plt.title("PCA of MFCC Features")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(
    title="Audio Files",
    loc="best",
    bbox_to_anchor=(1, 1),
    labels=[f"File {i+1}" for i in range(len(file_paths))],
)
plt.show()
