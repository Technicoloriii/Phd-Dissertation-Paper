import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets,decomposition
from mpl_toolkits.mplot3d import Axes3D
igt= pd.read_csv('D:\PhD PSU\Dissertation\Python project\data\igt.csv') # 本地加载数据

igt.columns
igt

igt=igt.dropna()
igt
type(igt)

igts=igt[["wapw","mhi","ur","lg(pl)","lg(gdp)","lg(ml)"]]

igts.columns
#igts = igts[np.isfinite(igts).all(1)]
igts
x=igts

y=igt.cl
y=y.values

pca=decomposition.PCA(n_components=None)
pca.fit(x)
print("explained variance ration: %s"% str(pca.explained_variance_ratio_))

pca=decomposition.PCA(n_components=2)
pca.fit(x)

X_r=pca.transform(x)
ny=y.reshape(309,1)

print(X_r.shape)
print(y.shape)
data=np.append(X_r,ny,axis=1)

data=pd.DataFrame(data)
data.columns=['PC1','PC2','Party']
print(data)

colors=['#850516','#0a39a1','#034f1a']
Party=data.Party.unique()
for i in range(len(Party)):
    plt.scatter(data.loc[data.Party==Party[i],"PC1"],data.loc[data.Party==Party[i],'PC2'],
               s=35,c=colors[i],label=Party[i])
plt.title("2-D Principle Components Plot")
plt.xlabel("1st Principle Component")
plt.ylabel("2nd Principle Conponent")
plt.legend(loc="upper left")
plt.savefig('D:\PhD PSU\Dissertation\Python project\pca',dpi=2000)
plt.show()