from sklearn import datasets
import matplotlib.pyplot as plt

digit=datasets.load_digits()


plt.figure(figsize=(5,5))
plt.imshow(digit.images[0],cmap=plt.cm.gray_r,interpolation='nearest') # 0번 샘플을 그림
plt.show()
print(digit.data[0]) # 0번 샘플의 화솟값을 출력
print("이 숫자는 ",digit.target[2],"입니다.")



