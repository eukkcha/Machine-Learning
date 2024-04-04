from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np
import math
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_palette("Set2")

x, y = make_blobs(n_samples=100, centers=4, n_features=2, random_state=6)
points = pd.DataFrame(x, y).reset_index(drop=True)
points.columns = ["x", "y"]
points.head()
print(points)

plt.figure()
plt.subplot(211)
sns.scatterplot(x="x", y="y", data=points, palette="Set2");



from sklearn.cluster import KMeans 

# k-means clustering 실행
kmeans = KMeans(n_clusters=4)
kmeans.fit(points)

# 결과 확인



result_by_sklearn = points.copy()
result_by_sklearn["cluster"] = kmeans.labels_
result_by_sklearn.head()

plt.subplot(212)
sns.scatterplot(x="x", y="y", hue="cluster", data=result_by_sklearn, palette="Set2");
plt.show()
