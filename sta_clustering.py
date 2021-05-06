# -*- coding: utf-8 -*-
"""
Created on Mon May 11 10:06:38 2020

@author: Admin
"""

import pandas as pd
from scipy.spatial import distance_matrix


#%%
data = [[5, 7], [7, 3], [8, 1]]
stas_dis=[]
with_sta_cells=np.zeros(len(esta_all),dtype=bool)
#valid_units = np.ones(ncl, dtype=bool)
high_snr=np.zeros(len(esta_all),dtype=bool)
mean_var_ratio=[]
for i in range(len(esta_all)):
    if not(esta_all[i]==None):
        if len(esta_all[i][2])>1:
            
#            zs_singl_esta=(esta_all[i][3]-mean(esta_all[i][3]))/std(esta_all[i][3],ddof=0).flatten()
#            high_snr[i]=abs(var(zs_singl_esta,axis=0)/np.average(zs_singl_esta,axis=0))<10000
           
            high_snr[i]=abs(var(esta_all[i][3],axis=0)/np.average(esta_all[i][3],axis=0))<5
            mean_var_ratio.append(abs(np.average(esta_all[i][3],axis=0)/var(esta_all[i][3],axis=0)))
#            mean_var_ratio.append(abs(var(zs_singl_esta,axis=0)/np.average(zs_singl_esta,axis=0)))
        
#        stas_dis.append(esta_all[i][3])
#           Cell_names_dist.append(Cells_names_and_clusters[i])
            with_sta_cells[i]=True



#high_snr=np.asarray(mean_var_ratio).flatten()>1
plt.hist(np.asarray(mean_var_ratio).flatten()            ,bins=55)
with_sta_cells=(high_snr & with_sta_cells )   



  
stas_3=np.asarray([row[3].flatten() for row in esta_all[with_sta_cells]])
#for i in stas_3
#    zssta=(stas_3-mean(stas_3))/std(stas_3,ddof=0)
zssta=[(row-mean(row))/std(row,ddof=0).flatten() for row in stas_3]
t=7
#ctys = [Cell_names_dist]
df = pd.DataFrame(zssta)
stas_dis=pd.DataFrame(distance_matrix(df.values, df.values))
#
#
#plt.imshow(stas_dis,origin='lower')
#plt.imshow(new_distances,origin='lower')
#
#new_stas_dis =stas_dis.reshape((n_remain,n_remain))
#aa = stas_dis[np.triu_indices_from(stas_dis,1)]
#
#new_distances = stas_dis.reshape((n_remain,n_remain))
#ISI_dist_ys_valid[i] = new_distances[np.triu_indices_from(new_distances,1)]

from scipy.spatial.distance import pdist

l = cluster.hierarchy.linkage(zssta, "ward")
#cluster.hierarchy.dendrogram(l);

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
silhouettes = metrics.silhouette_samples(distance.squareform(pdist(stas_3)),fcls,metric='precomputed')
print("Mean Silhouette Coefficient: %.2f" % np.average(silhouettes))

n_flat_clusters = np.unique(fcls).shape[0]
n = n_flat_clusters
T = np.unique(fcls)
labels=list('' for i in range(20*n))
for i in range(n):
    labels[i]=str(i)+ ',' + str(T[i])
show_order = np.unique(fcls)[::-1]-1
max_num_clusters = np.unique(fcls).shape[0]

palette = sns.hls_palette(max_num_clusters,l=0.6,s=0.6)    

max_num_clustersx=t

# plt.figure(figsize=(9,max_num_clustersx*1.7/4.8))
plt.figure(figsize=(16,max_num_clustersx*1))


fs = 12#24

labels=list('' for i in range(n))
for i in range(n):
    labels[i] = '#' + str(T[i]) + '\n(' + str(np.count_nonzero(np.where(fcls==T[i]))) + ')'
    
# plot_widths = [1.2,0.8,3,0.4,0.4, 0.4, 0.4]
plot_widths = [1,2,1,6.4,0.8, 0.4, 0.3]

gs = gridspec.GridSpec(max_num_clustersx, 7, width_ratios = plot_widths, wspace=0.1, hspace=0.1)

bins = 20 # ms
Cells_names_and_clusters=list(zip(Cell_names[with_sta_cells],fcls))

cids = np.where(conditions_all)[0]


for ii, c in enumerate(show_order):
    
    
    n_units = np.where(fcls == c+1)[0].shape[0]
    for ci, stimid in enumerate(plot_stims):
        sts = Stimuli['SpikeTrains'][stimid][with_sta_cells][np.where(fcls == c+1)]
        if stimid == 0:
            txt = "Cluster %d (%d units)" % (c+1,n_units)
        elif stimid == 1:
            t_sils = silhouettes[np.where(fcls == c+1)]
            t_ooi  = OOi[with_sta_cells][np.where(fcls == c+1)]
            txt = "Avg. OOi: %.2f±%.2f    Avg. Silhouette Coeff.: %.2f±%.2f" % (np.average(t_ooi),np.std(t_ooi),np.average(t_sils),np.std(t_sils))
        elif stimid == 5:
            t_dsi = DSi[with_sta_cells][np.where(fcls == c+1)][:,0]
            txt = "Avg. DSi: %.2f±%.2f" % (np.average(t_dsi),np.std(t_dsi))
        else:
            txt = ""
        if (ci == 0) & (ii==0):
            txt = 'B'
        elif (ci == 1) & (ii==0):
            txt = 'C'
        else:
            txt = None
        with plt.rc_context({'font.size':fs, 'axes.titleweight': 'bold'}):
            ax = plt.subplot(gs[ii,ci+2])
            plotPSTHs(ax,sts,txt,bins,palette[c], show_sd=False, lw=1)
            plt.xticks(())
            plt.yticks(())
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
    with plt.rc_context({'font.size':fs, 'axes.titleweight': 'bold'}):
        ax = plt.subplot(gs[ii,4])
        t_dsi = DSi[with_sta_cells][np.where(fcls == c+1)][:,0]
        sns.distplot(t_dsi,bins=np.arange(0,0.8,0.1),kde=True, norm_hist=True)
        if ii == 0:
            txt = 'D'
        else:
            txt = ''
        t_dsi = np.median(DSi[with_sta_cells][:,0])
        plt.plot((t_dsi, t_dsi),ax.get_ylim(),'r')
        ax.set_title(txt)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_xlim((0,1.0))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if ii==len(show_order)-1:
            plt.xlabel('DSI', fontsize=8)
            

    
    
    
    with plt.rc_context({'font.size':fs, 'axes.titleweight': 'bold'}):
        ax = plt.subplot(gs[ii,1])
#        t_dsi = DSi[conditions_all][np.where(fcls == c+1)][:,0]
#        sns.distplot(t_dsi,bins=np.arange(0,0.8,0.1),kde=True, norm_hist=True)
        if ii == 0:
            txt = 'E'
        else:
            txt = ''
 
        trl_sta=[]    
        stas=esta_all[with_sta_cells][np.where(fcls == c+1)]
        for i in range(len(stas)):
            if  not(stas[i]==None):
                if len(stas[i][2])>1:
                    stime=stas[i][2]
                    ssta=stas[i][3]
    
                    xnew = np.linspace(stime.min(), stime.max(), 200)  
                    
                    sta_smooth = spline(stime.flatten(), ssta.flatten(), xnew)
                    sta_smooth=(sta_smooth-mean(sta_smooth))/std(sta_smooth,ddof=0)
                    trl_sta.append(sta_smooth)
                    
                    plt.plot(xnew,sta_smooth,linewidth=.8)
                    plt.axhline(y=0, color='k', linestyle='--',linewidth=.8)
#                    ax.set_ylim((stime.min(),stime.max()))
                    plt.axvline(x=.020, color='k', linestyle='--',linewidth=.8)
                
#                plt.plot(stas[i][2],(stas[i][3]-mean(stas[i][3]))/max(abs(stas[i][3])))
        if trl_sta:
            plt.plot(xnew,mean(trl_sta,axis=0),linewidth=1.5,color='k')
            plt.text(.2,1.2,str(" %.2f" % (median(var(trl_sta,axis=0))/median(np.average(trl_sta,axis=0)))))
            #plt.text(.7,.9,str(" %.2f" % var(var(trl_sta,axis=0))))

        ax.set_title(txt)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_xlim((-1,0.5))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        alpha =.5
        if ii==len(show_order)-1:
            plt.xlabel('esta', fontsize=8)    
            
            
    ax = plt.subplot(gs[:,0])
    p = ax.get_position()
    p.x1 = p.x1-0.0025
    ax.set_position(p)
    with plt.rc_context({'lines.linewidth': 2, 'font.size':fs, 'axes.titleweight': 'bold'}):
        dend = cluster.hierarchy.dendrogram(l, p=n, no_labels=False, leaf_font_size=15, color_threshold=t, 
                                            distance_sort='none', link_color_func=lambda k: cs[k-l.shape[0]-1], 
                                            truncate_mode='lastp', show_leaf_counts=True, orientation='left')
    
        ax.set_title("A")
        ax.set_yticklabels(labels)
        ax.set_xticks(())
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
if SAVE_FIGS:
    plt.savefig('clusters_e-STA.pdf', bbox_inches='tight')
        