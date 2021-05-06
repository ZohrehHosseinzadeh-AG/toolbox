# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 11:09:58 2021

@author: Admin
"""


c=3
celnmbrs=np.where(fcls_sta == c+1)
t_ooi  = OOi[conditions_all][np.where(fcls_sta == c+1)]
txt=" %.2f   " % (np.mean(t_ooi))

i=0
stimid=0
bidex=True
rownmbr=size(celnmbrs)
gs = gridspec.GridSpec(rownmbr+1, 1, wspace=0.1, hspace=0.1)
plt.figure(figsize=(4,rownmbr))


for i,cellss in enumerate(np.nditer(celnmbrs)):
    #print(cellss)
    if i>0:
        txt=''
    sts = Stimuli['SpikeTrains'][stimid][conditions_all][cellss]
    with plt.rc_context({'font.size':fs, 'axes.titleweight': 'bold'}):

        ax = plt.subplot(gs[i,0])
        plotPSTHs(ax,sts,txt,bins,palette[c],show_ticks=True, show_sd=False, lw=1)

        
        if bidex == True:            
            xy,ys = spk.psth(sts.flatten(), 20).get_plottable_data()
            ys = ys / sts.shape[0]
            plt.text(2500, .7*max(ys), " %.2f   " % (OOi[conditions_all][cellss]), fontsize=9)
            
        plt.xticks(())
        #plt.yticks(())
        
ax = plt.subplot(gs[i+1,0])
stsall= Stimuli['SpikeTrains'][stimid][conditions_all][np.where(fcls_sta == c+1)]
plotPSTHs(ax,stsall,'Average of all',bins,palette[c],show_ticks=True, show_sd=False, lw=1)
plt.text(2500, .7*max(ys), " %.2f   " % (np.mean(OOi[conditions_all][np.where(fcls_sta == c+1)])), fontsize=9)
