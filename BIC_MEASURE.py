# -*- coding: utf-8 -*-
"""
Created on Tue May 19 09:38:21 2020

@author: Admin
"""
stas_4=np.asarray([row[3].flatten() for row in esta_all[min_maz>10]])
plot(stas_4[:].T)
stas_5=np.asarray([row[3].flatten() for row in esta_all[(min_maz<20) & (min_maz>10)]])
plot(stas_5[:].T,alpha=.8)
#%%
np.where(esta_all[min_maz>1])
#%%
plot(stas_4[:].T)
#%%
ii=1
idx=np.where(with_sta_cells)
idxx=idx[0][ii]
plot(esta_all[idxx][3])

#%%

last = l[-10:, 2]
last_rev = last[::-1]
idxs = np.arange(1, len(last) + 1)
plt.plot(idxs, last_rev)

acceleration = np.diff(last, 2)  # 2nd derivative of the distances
acceleration_rev = acceleration[::-1]
plt.plot(idxs[:-2] + 1, acceleration_rev)
plt.show()
k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
print("clusters:", k)
#%%
l = cluster.hierarchy.linkage(zssta, "ward")
#cluster.hierarchy.dendrogram(l);
silhouettess=[]
for kk in range(2,29):
    t=kk
    #l = cluster.hierarchy.linkage(zssta, method='average')
    #fcls = cluster.hierarchy.fcluster(l, t=9.70909, criterion='distance')
    fcls = cluster.hierarchy.fcluster(l, t=t, criterion='maxclust')
    #kmeans = KMeans(n_clusters=20, random_state=0).fit(zssta)
    #fcls=kmeans.labels_
    
    #fcls = cluster.hierarchy.fcluster(l, 1.80, depth=10)
    n = 41
    pal = sns.diverging_palette(180,359,sep=1,n=n)
    OOi_cspace = np.linspace(-1,1,n)
    OOi_c_func = lambda i: pal[np.searchsorted(OOi_cspace,OOi[with_sta_cells][i])]
    DSi_cspace = np.linspace(0,1,n)
    DSi_c_func = lambda i: pal[np.searchsorted(DSi_cspace,DSi[with_sta_cells][i,0])]
    
    cs = create_colors_for_linkage(l,l.shape[0]+1,OOi_c_func)
    
    c, coph_dists = cluster.hierarchy.cophenet(l, pdist(zssta))
    c
    silhouettes= metrics.silhouette_samples(distance.squareform(pdist(zssta)),fcls,metric='precomputed')
    silhouettess.append(np.median(silhouettes))
    print("Mean Silhouette Coefficient: %.2f" % np.median(silhouettes))
plot(silhouettess)
#%%
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_scaled=scaler.fit_transform(stas_3)
X_scaled=np.asarray(zssta)
from sklearn.mixture import GaussianMixture

gm_bic= []
gm_score=[]
for i in range(2,19):
    gm = GaussianMixture(n_components=i,n_init=10,tol=1e-3,max_iter=1000).fit(X_scaled)
    print("BIC for number of cluster(s) {}: {}".format(i,gm.bic(X_scaled)))
    print("Log-likelihood score for number of cluster(s) {}: {}".format(i,gm.score(X_scaled)))
    print("-"*100)
    gm_bic.append(gm.bic(X_scaled))
    gm_score.append(gm.score(X_scaled))
    
plt.figure(figsize=(7,4))
plt.title("The Gaussian Mixture model BIC \nfor determining number of clusters\n",fontsize=16)
plt.scatter(x=[i for i in range(2,19)],y=np.log(gm_bic),s=150,edgecolor='k')
plt.grid(True)
plt.xlabel("Number of clusters",fontsize=14)
plt.ylabel("Log of Gaussian mixture BIC score",fontsize=15)
plt.xticks([i for i in range(2,19)],fontsize=14)
plt.yticks(fontsize=15)
plt.show()    
plt.scatter(x=[i for i in range(2,19)],y=gm_score,s=150,edgecolor='k')
plt.show()