import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import librosa

# 오디오 파일을 로드하고 MFCC 특성 추출
def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)  # 오디오 파일 로드
    mfcc = librosa.feature.mfcc(audio, sr, n_mfcc=35, hop_length = 1024)  # MFCC 특성 추출
    return mfcc.T

# 오디오 파일 경로 목록
file_paths = [
    'wavfile/규보_[cut_8sec].wav',
    'wavfile/아빠_[cut_8sec].wav',
    'wavfile/엄마_[cut_8sec].wav',
    'wavfile/유성_[cut_8sec].wav',
    'wavfile/유찬_[cut_8sec].wav',
    'wavfile/이모_[cut_8sec].wav',
    'wavfile/지윤_[cut_8sec].wav',
    'wavfile/창희_[cut_8sec].wav',
    'wavfile/할머니_[cut_8sec].wav',
    'wavfile/형_[cut_8sec].wav'
]

models = []  # GMM 모델들을 저장할 리스트
all_features = []  # 모든 MFCC 특성들을 저장할 리스트
labels = []  # 각 파일의 라벨을 저장할 리스트

# 각 오디오 파일에 대해 GMM 모델을 학습하고 특성을 저장
for i, file_path in enumerate(file_paths):
    mfcc_features = load_audio(file_path)  # MFCC 특성 추출
    gmm = GaussianMixture(n_components=2, covariance_type='diag', random_state = 0)  # GMM 모델 초기화
    gmm.fit(mfcc_features)  # GMM 모델 학습
    models.append(gmm)  # 학습된 GMM 모델 저장
    all_features.append(mfcc_features)  # 추출된 MFCC 특성 저장
    labels.extend([i] * len(mfcc_features))  # 각 파일에 대한 라벨 저장

all_features = np.vstack(all_features)  # 모든 MFCC 특성을 하나의 배열로 결합
labels = np.array(labels)  # 라벨을 배열로 변환

# 테스트 음성 파일 경로 목록
test_file_paths = [
    'wavfile/규보_[cut_3sec].wav',
    'wavfile/아빠_[cut_3sec].wav',
    'wavfile/엄마_[cut_3sec].wav',
    'wavfile/유성_[cut_3sec].wav',
    'wavfile/유찬_[cut_3sec].wav',
    'wavfile/이모_[cut_3sec].wav',
    'wavfile/지윤_[cut_3sec].wav',
    'wavfile/창희_[cut_3sec].wav',
    'wavfile/할머니_[cut_3sec].wav',
    'wavfile/형_[cut_3sec].wav'
]

# 혼돈 행렬 초기화
conf = np.zeros((10, 10))

# 각 테스트 파일에 대해 예측을 수행하고 혼돈 행렬 갱신
individual_accuracies = []
for y_test, test_file_path in enumerate(test_file_paths):
    test_features = load_audio(test_file_path)  # 테스트 파일의 MFCC 특성 추출
    correct_predictions = 0
    for feature in test_features:
        scores = [model.score(feature.reshape(1, -1)) for model in models]
        predicted_label = np.argmax(scores)
        conf[predicted_label][y_test] += 1
        if predicted_label == y_test:
            correct_predictions += 1
    accuracy = correct_predictions / len(test_features) * 100
    individual_accuracies.append(accuracy)
    print(f'Accuracy for Model {y_test + 1} with Test {y_test + 1}: {accuracy:.2f}%')

print(conf)

# 전체 정확률 측정하고 출력
no_correct = np.trace(conf)
overall_accuracy = no_correct / np.sum(conf) * 100
print(f'Overall Accuracy: {overall_accuracy:.2f}%')

# 시각화를 위해 차원 축소
pca = PCA(n_components=2)
features_pca = pca.fit_transform(all_features)  # PCA를 이용해 2차원으로 축소

# 서브플롯 설정
fig, axs = plt.subplots(5, 2, figsize=(20, 20))
axs = axs.ravel()  # 2D 배열을 1D 배열로 변환

# 각 테스트 파일과 모델을 서브플롯으로 시각화
for y_test, (test_file_path, ax) in enumerate(zip(test_file_paths, axs)):
    test_features = load_audio(test_file_path)
    test_features_pca = pca.transform(test_features)  # 테스트 파일의 특성도 동일한 PCA 변환 적용
    
    model_features_pca = features_pca[labels == y_test]  # 해당 모델의 특성만 선택

    ax.scatter(model_features_pca[:, 0], model_features_pca[:, 1], label=f'Model {y_test+1}', alpha=0.5)
    ax.scatter(test_features_pca[:, 0], test_features_pca[:, 1], color='black', marker='x', label=f'Test {y_test+1}', alpha=0.8)

    ax.set_xlabel('x')  # x축 레이블
    ax.set_ylabel('y')  # y축 레이블
    ax.set_title(f'Model {y_test+1} + Test {y_test+1}\nAccuracy: {individual_accuracies[y_test]:.2f}%')  # 서브플롯 제목
    ax.legend()  # 범례 추가

plt.tight_layout()  # 서브플롯 간의 레이아웃 조정
plt.show()  # 그래프 출력
