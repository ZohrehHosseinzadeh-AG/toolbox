# -*- coding: utf-8 -*-
"""
Created on Mon May 11 10:06:38 2020

@author: Admin
"""

import pandas as pd
from scipy.spatial import distance_matrix

half_sta=int(len(esta_i)/2+1)
baseline_sta = np.average(esta_i[half_sta:]);
baseline_std = np.std(esta_i[half_sta:]);
alphaville=.001
import statsmodels.api as sm
s_,p=sm.stats.ztest(esta_i[half_sta:], x2=esta_i[0:half_sta], value=0, alternative='two-sided', ddof=1.0)

from scipy import stats
np.random.seed(28041990)
pts = 1000
a = np.random.normal(2, 1, size=pts)
b = np.random.normal(2, 1, size=pts)
x = np.concatenate((a, b))
k2, p = stats.normaltest(esta_i)
p
#%%
stas_dis=[]
with_sta_cells=np.zeros(len(esta_all),dtype=bool)
#valid_units = np.ones(ncl, dtype=bool)
high_snr=np.zeros(len(esta_all),dtype=bool)
p_values=np.zeros(len(esta_all),dtype=bool)
mean_var_ratio=[]
min_maz=np.zeros(len(esta_all))
for i in range(len(esta_all)):
    if not(esta_all[i]==None):
        if len(esta_all[i][2])>1:
            mean_var_ratio.append(abs(np.average(esta_all[i][3],axis=0)/var(esta_all[i][3],axis=0)))
            with_sta_cells[i]=True
            peaks=np.hstack((np.argmax(esta_i),np.argmax(-esta_i)))
            peak_sta=esta_i[peaks[np.argmax([np.linalg.norm((esta_i[peaks][0]-mean(esta_i))),np.linalg.norm((esta_i[peaks][1]-mean(esta_i)))])]]
            pck=peaks[np.argmax(np.linalg.norm((esta_i[peaks],mean(esta_i))))]
            esta_ii=esta_i[24:]#np.delete(esta_i, pck)
            high_snr[i]=np.linalg.norm(peak_sta-mean(esta_i))>4*std(esta_ii)
            k2, p = stats.normaltest(esta_i)
            p_values[i]=p<.001

plt.hist(np.asarray(mean_var_ratio).flatten(), bins=55);
with_sta_cells=( (p_values)&(with_sta_cells) )   
stas_3=np.asarray([row[3].flatten() for row in esta_all[with_sta_cells]])
zssta=[(row-mean(row))/std(row,ddof=0).flatten() for row in stas_3]

df = pd.DataFrame(zssta)
stas_dis=pd.DataFrame(distance_matrix(df.values, df.values))

from scipy.spatial.distance import pdist
l = cluster.hierarchy.linkage(zssta, "ward")
t=6
fcls = cluster.hierarchy.fcluster(l, t=t, criterion='maxclust')
n = 41
pal = sns.diverging_palette(180,359,sep=1,n=n)
OOi_cspace = np.linspace(-1,1,n)
OOi_c_func = lambda i: pal[np.searchsorted(OOi_cspace,OOi[with_sta_cells][i])]
DSi_cspace = np.linspace(0,1,n)
DSi_c_func = lambda i: pal[np.searchsorted(DSi_cspace,DSi[with_sta_cells][i,0])]

cs = create_colors_for_linkage(l,l.shape[0]+1,OOi_c_func)

c, coph_dists = cluster.hierarchy.cophenet(l, pdist(zssta))
c
silhouettes = metrics.silhouette_samples(distance.squareform(pdist(zssta)),fcls,metric='precomputed')
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
plt.figure(figsize=(14,max_num_clustersx*1))
# plt.figure(figsize=(9,max_num_clustersx*1.7/4.8))

plot_stims = [0,1,10]

fs = 12#24

labels=list('' for i in range(n))
for i in range(n):
    labels[i] = '#' + str(T[i]) + '\n(' + str(np.count_nonzero(np.where(fcls==T[i]))) + ')'
    
# plot_widths = [1.2,0.8,3,0.4,0.4, 0.4, 0.4]
plot_widths = [2,2.8,1.3,7.5,2.3,.7, 0.6]

gs = gridspec.GridSpec(max_num_clustersx, 7, width_ratios = plot_widths, wspace=0.1, hspace=0.1)

bins = 20 # ms

cids = np.where(with_sta_cells)[0]
has_sta = np.zeros_like(with_sta_cells).astype(dtype=bool)
#has_sta[STAs['units'].value] = True
sta_inds = np.zeros(np.sum(with_sta_cells), dtype(int))
#sta_inds[has_sta[with_sta_cells]] = np.where(np.isin(STAs['units'].value, np.where(with_sta_cells&has_sta)[0]))[0]
has_sta = has_sta[with_sta_cells]
#mean_rf_size = np.median((np.abs(STAs['fits'][has_sta,3]),np.abs(STAs['fits'][has_sta,3])))
mean_rf_size=np.zeros(np.sum(with_sta_cells), dtype(int))
ylims = [(0,40),(0,10)]
    
for i, c in enumerate(show_order):

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
            txt = "E"
        if (ci == 0) & (i==0):
            txt = 'C'
        elif (ci == 1) & (i==0):
            txt = 'D'
        elif (ci == 2) & (i==0):
            txt = 'E'
        else:
            txt = None
        with plt.rc_context({'font.size':fs, 'axes.titleweight': 'bold'}):
            ax = plt.subplot(gs[i,ci+2])
            plotPSTHs(ax,sts,txt,bins,palette[c], show_sd=False, lw=1)
            plt.xticks(())
            plt.yticks(())
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)

        
    with plt.rc_context({'font.size':fs, 'axes.titleweight': 'bold'}):
        ax = plt.subplot(gs[i,5])
        t_dsi = DSi[with_sta_cells][np.where(fcls == c+1)][:,0]
        sns.distplot(t_dsi,bins=np.arange(0,0.8,0.1),kde=True, norm_hist=True)
        if i == 0:
            txt = 'F'
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
        if i==len(show_order)-1:
            plt.xlabel('DSI', fontsize=8)
            
            
    with plt.rc_context({'font.size':fs, 'axes.titleweight': 'bold'}):
        ax = plt.subplot(gs[i,1])
#        t_dsi = DSi[with_sta_cells][np.where(fcls == c+1)][:,0]
#        sns.distplot(t_dsi,bins=np.arange(0,0.8,0.1),kde=True, norm_hist=True)
        if i == 0:
            txt = 'B'
        else:
            txt = ''
#        t_dsi = np.median(DSi[with_sta_cells][:,0])
        trl_sta=[]    
        stas=esta_all[with_sta_cells][np.where(fcls == c+1)]
        for ii in range(len(stas)):
            if  not(stas[ii]==None):
                if len(stas[ii][2])>1:
                    stime=stas[ii][2]
                    ssta=stas[ii][3]
    
                    xnew = np.linspace(stime.min(), stime.max(), 200)  
                    
                    sta_smooth = spline(stime.flatten(), ssta.flatten(), xnew)
                    sta_smooth=(sta_smooth-mean(sta_smooth))/std(sta_smooth,ddof=0)
                    trl_sta.append(sta_smooth)
                    
                    plt.plot(xnew,sta_smooth,linewidth=.8,alpha=.7)
                    plt.axhline(y=0, color='k', linestyle='--',linewidth=.8)
#                    ax.set_ylim((stime.min(),stime.max()))
                    plt.axvline(x=.020, color='k', linestyle='--',linewidth=.8)
                
#                plt.plot(stas[i][2],(stas[i][3]-mean(stas[i][3]))/max(abs(stas[i][3])))
        if trl_sta:
            plt.plot(xnew,mean(trl_sta,axis=0),linewidth=1.5,color='k')
        ax.set_title(txt)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_xlim((-1,0.5))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        alpha =.5
        if i==len(show_order)-1:
            plt.xlabel('esta', fontsize=8)    
#    with plt.rc_context({'axes.facecolor': 'grey','axes.edgecolor': 'grey','font.size':fs, 'axes.titleweight': 'bold'}):
#        ax = plt.subplot(gs[i,3])
#        n_units = np.where(fcls == c+1)[0].shape[0]
#        ax.scatter(ClusterLoc[0][0,with_sta_cells][np.where(fcls != c+1)],
#                   ClusterLoc[0][1,with_sta_cells][np.where(fcls != c+1)], c='w', s=3)
#        ax.scatter(ClusterLoc[0][0,with_sta_cells][np.where(fcls == c+1)],
#                   ClusterLoc[0][1,with_sta_cells][np.where(fcls == c+1)], s=3, c=palette[c])
#        if i == 0:
#            txt = 'D'
#        else:
#            txt = ''
#        ax.set_title(txt)
#        ax.set_xlim((-1,65))
#        ax.set_ylim((-1,65))
#        ax.set_xticks(())
#        ax.set_yticks(())
#        ax.set_aspect(1)

#    with plt.rc_context({'font.size':fs, 'axes.titleweight': 'bold'}):
#        inds = fcls==c+1
#        usable_stas = sta_inds[has_sta&inds]
#        if len(usable_stas)>0:
#            ax = plt.subplot(gs[i,5])
#            sx = np.mean((np.abs(STAs['fits'][usable_stas,3]),np.abs(STAs['fits'][usable_stas,4])),0)
#            sns.distplot(sx[(sx<10)&(sx>0)],bins=np.arange(0,6,0.5),kde=True, norm_hist=True)
#            plt.plot((mean_rf_size,mean_rf_size),ax.get_ylim(),'r')
#            if i == 0:
#                txt = 'F'
#            else:
#                txt = ''
#            ax.set_title(txt)
#            ax.set_xticks(())
#            ax.set_yticks(())
#            ax.set_xlim((1,6))
#            ax.spines['top'].set_visible(False)
#            ax.spines['right'].set_visible(False)
#        if i==len(show_order)-1:
#            plt.xlabel('RF size', fontsize=8)

#    with plt.rc_context({'font.size':fs, 'axes.titleweight': 'bold'}):
#        ax = plt.subplot(gs[i,6])
#        if len(usable_stas)>0:
##            tmp_sta = np.array([get_sta_at_peak(STAs['STAs'][i]) for i in usable_stas])
#            x = np.arange(tmp_sta.shape[1])
#            plt.fill_between(x, np.mean(tmp_sta,0)-np.std(tmp_sta,0), np.mean(tmp_sta,0)+np.std(tmp_sta,0), color=palette[c])
#            plt.plot(x,np.mean(tmp_sta,0),'k')
#        if i == 0:
#            txt = 'G'
#        else:
#            txt = ''
#        ax.set_title(txt)
#        ax.set_xticks(())
#        ax.set_yticks(())
#        plt.ylim((-0.3,0.3))
#        ax.spines['top'].set_visible(False)
#        ax.spines['bottom'].set_visible(False)
#        ax.spines['left'].set_visible(False)
#        ax.spines['right'].set_visible(False)
        
ax = plt.subplot(gs[:,0])
p = ax.get_position()
p.x1 = p.x1-0.02
ax.set_position(p)
with plt.rc_context({'lines.linewidth': 2, 'font.size':fs, 'axes.titleweight': 'bold'}):
    dend = cluster.hierarchy.dendrogram(l, p=n, no_labels=False, leaf_font_size=7, color_threshold=t, 
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
   plt.savefig('clusters_e_STA_color_mixedrd.pdf', bbox_inches='tight')
   plt.savefig('clusters_e_STA_color_mixedrd.png', bbox_inches='tight')

    
    
    
    
Cells_names_and_clusters_sta=list(zip(Cell_names[with_sta_cells],fcls))
    
with open("clusetr_numbers_sta.txt", 'w') as output:
    for row in Cells_names_and_clusters_sta:
        output.write(str(row) + '\n')      
