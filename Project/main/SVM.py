import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA
import librosa
import seaborn as sns


# 오디오 파일을 로드하고 MFCC 특성 추출
def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=44100)  # 오디오 파일 로드
    mfcc = librosa.feature.mfcc(audio, sr, n_mfcc=108)  # MFCC 특성 추출
    return mfcc.T


# 오디오 파일 경로 목록
file_paths = [
    "wav4/규보4.wav",
    "wav4/아빠4.wav",
    "wav4/엄마4.wav",
    "wav4/유상4.wav",
    "wav4/유성4.wav",
    "wav4/유찬4.wav",
    "wav4/이모4.wav",
    "wav4/지윤4.wav",
    "wav4/창희4.wav",
    "wav4/할머니4.wav"
]

all_features = []  # 모든 MFCC 특성들을 저장할 리스트
labels = []  # 각 파일의 라벨을 저장할 리스트

# 각 오디오 파일에 대해 MFCC 특성을 저장
for i, file_path in enumerate(file_paths):
    mfcc_features = load_audio(file_path)  # MFCC 특성 추출
    all_features.append(mfcc_features)  # 추출된 MFCC 특성 저장
    labels.extend([i] * len(mfcc_features))  # 각 파일에 대한 라벨 저장

all_features = np.vstack(all_features)  # 모든 MFCC 특성을 하나의 배열로 결합
labels = np.array(labels)  # 라벨을 배열로 변환

# SVM 모델 초기화 및 학습
svm_model = svm.SVC(gamma="scale", probability=True)
svm_model.fit(all_features, labels)

# 테스트 음성 파일 경로 목록
test_file_paths = [
    "wav4/규보123.wav",
    "wav4/아빠123.wav",
    "wav4/엄마123.wav",
    "wav4/유상123.wav",
    "wav4/유성123.wav",
    "wav4/유찬123.wav",
    "wav4/이모123.wav",
    "wav4/지윤123.wav",
    "wav4/창희123.wav",
    "wav4/할머니123.wav"
]

# 혼돈 행렬 초기화
conf = np.zeros((10, 10))

# 각 테스트 파일에 대해 예측을 수행하고 혼돈 행렬 갱신
individual_accuracies = []
for y_test, test_file_path in enumerate(test_file_paths):
    test_features = load_audio(test_file_path)  # 테스트 파일의 MFCC 특성 추출
    correct_predictions = 0
    for feature in test_features:
        predicted_label = svm_model.predict(feature.reshape(1, -1))[0]
        conf[predicted_label][y_test] += 1
        if predicted_label == y_test:
            correct_predictions += 1
    accuracy = correct_predictions / len(test_features) * 100
    individual_accuracies.append(accuracy)
    print(f"Accuracy for Model {y_test + 1} with Test {y_test + 1}: {accuracy:.2f}%")

print(conf)

# 전체 정확률 측정하고 출력
no_correct = np.trace(conf)
overall_accuracy = no_correct / np.sum(conf) * 100
print(f"Overall Accuracy: {overall_accuracy:.2f}%")

# 혼돈 행렬 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(
    conf,
    annot=True,
    fmt="g",
    cmap="Blues",
    xticklabels=[f"Test {i+1}" for i in range(10)],
    yticklabels=[f"Model {i+1}" for i in range(10)],
)
plt.xlabel("Actual Label")
plt.ylabel("Predicted Label")
plt.title("Confusion Matrix")
plt.show()

# 시각화를 위해 차원 축소
pca = PCA(n_components=2)
features_pca = pca.fit_transform(all_features)  # PCA를 이용해 2차원으로 축소

# 서브플롯 설정
fig, axs = plt.subplots(5, 2, figsize=(20, 20))
axs = axs.ravel()  # 2D 배열을 1D 배열로 변환

# 각 테스트 파일과 모델을 서브플롯으로 시각화
for y_test, (test_file_path, ax) in enumerate(zip(test_file_paths, axs)):
    test_features = load_audio(test_file_path)
    test_features_pca = pca.transform(
        test_features
    )  # 테스트 파일의 특성도 동일한 PCA 변환 적용

    model_features_pca = features_pca[labels == y_test]  # 해당 모델의 특성만 선택

    ax.scatter(
        model_features_pca[:, 0],
        model_features_pca[:, 1],
        label=f"Model {y_test+1}",
        alpha=0.5,
    )
    ax.scatter(
        test_features_pca[:, 0],
        test_features_pca[:, 1],
        color="black",
        marker="x",
        label=f"Test {y_test+1}",
        alpha=0.8,
    )

    ax.set_xlabel("x")  # x축 레이블
    ax.set_ylabel("y")  # y축 레이블
    ax.set_title(
        f"Model {y_test+1} + Test {y_test+1}\nAccuracy: {individual_accuracies[y_test]:.2f}%"
    )  # 서브플롯 제목
    ax.legend()  # 범례 추가

plt.tight_layout()  # 서브플롯 간의 레이아웃 조정
plt.show()  # 그래프 출력
