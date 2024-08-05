import sklearn
import matplotlib.pyplot as plt
import sklearn.cluster
import sklearn.datasets


list_color = ["red","green","blue","black"]
set_cluster = 4
train_data,_ = sklearn.datasets.make_blobs(n_samples=300,centers=set_cluster,cluster_std=0.5,random_state=0)

test_data ,_ = sklearn.datasets.make_blobs(n_samples=20,centers=set_cluster,cluster_std=0.5,random_state=0)

K_mean_clusterong_modal = sklearn.cluster.KMeans(n_clusters=set_cluster)
K_mean_clusterong_modal.fit(X=train_data)

centroid = K_mean_clusterong_modal.cluster_centers_
print(centroid)

y_pred = K_mean_clusterong_modal.predict(X=train_data)
y_pred_test = K_mean_clusterong_modal.predict(X=test_data)

plt.figure(figsize=(16,10))

plt.subplot(1,2,1)
plt.scatter(train_data[:,0],y=train_data[:,1],c=y_pred,label="init_data")
plt.scatter(test_data[:,0],y=test_data[:,1],label="test_data")

for i in range(set_cluster):
    plt.scatter(centroid[i,0],centroid[i,1],c=list_color[i],label= f"Centroid {i}",s=120)

plt.legend(frameon=True)

plt.subplot(1,2,2)
plt.scatter(train_data[:,0],y=train_data[:,1],c=y_pred,label="init_data")
plt.scatter(test_data[:,0],y=test_data[:,1],c=y_pred_test,label="test_data")

for i in range(set_cluster):
    plt.scatter(centroid[i,0],centroid[i,1],c=list_color[i],label= f"Centroid {i}",s=120)

plt.legend(frameon=True)
plt.show()
