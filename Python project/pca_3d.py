import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn import datasets,decomposition


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

y=igt.p
y=y.values

pca=decomposition.PCA(n_components=None)
pca.fit(x)
print("explained variance ration: %s"% str(pca.explained_variance_ratio_))

pca=decomposition.PCA(n_components=3)
pca.fit(x)

X_r=pca.transform(x)
ny=y.reshape(309,1)

print(X_r.shape)
print(y.shape)
data=np.append(X_r,ny,axis=1)

data=pd.DataFrame(data)
data.columns=['PC1','PC2','PC3','Party']
print(data)

colors=['#850516','#0a39a1','#034f1a']
######2D图像展示############
# Party=data.Party.unique()
# for i in range(len(Party)):
#     plt.scatter(data.loc[data.Party==Party[i],"PC1"],data.loc[data.Party==Party[i],'PC2'],
#                s=35,c=colors[i],label=Party[i])
# plt.title("2-D Principle Components Plot")
# plt.xlabel("1st Principle Component")
# plt.ylabel("2nd Principle Conponent")
# plt.legend(loc="upper left")
# plt.savefig('D:\PhD PSU\Dissertation\Python project\pca_1',dpi=600)
# plt.show()

#####3D##########图像展示
# Creating figures for the plot
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
# Creating a plot using the random datasets
Party=data.Party.unique()
for i in range(len(Party)):
    ax.scatter3D(data.loc[data.Party==Party[i],"PC1"],data.loc[data.Party==Party[i],'PC2'],
                s=35,c=colors[i],label=Party[i])
plt.title("3D Principle Components Scatter Plot")
ax.set_xlabel('Principle Component 1', fontweight ='bold')
ax.set_ylabel('Principle Component 2', fontweight ='bold')
ax.set_zlabel('Principle Component 3', fontweight ='bold')
# display the  plot
plt.savefig('D:\PhD PSU\Dissertation\Python project\pca_3d',dpi=2000)
plt.show()