from sklearn.decomposition import PCA
import numpy as np
cv2.waitKey(0)
X=np.array([[4,9,7,-4],[5,9,-65,68],[16,2,38,108],[59,48,72,12],[66,45,89,70],[43,12,56,33]])
pca=PCA(n_components=2)
pca.fit(X)
print(pca.fit_transform(X))
