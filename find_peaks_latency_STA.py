# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 12:03:35 2021

@author: Admin
"""
import numpy as np
from scipy.interpolate import make_interp_spline
#Mean_sta=Mean_sta_visual
#plot(goodstas[0][3])
def find_peaks_latency_STA(Mean_sta,stime,figname):
    tabel=[]
    upsampling=5
    binlenght=40/upsampling
    shift=5
    upshift=upsampling*shift
    for c in range(len(Mean_sta)):
        len_sta=int(len(Mean_sta[0])/2+shift)
        rsta=Mean_sta[c][:len_sta].flatten()# includes 100 ms before zero
        xnew = np.linspace(stime[:len_sta].min(), stime[:len_sta].max(), upsampling*len(rsta)) # five times upsampling
                    
        sta_smooth = make_interp_spline(stime[:len_sta].flatten(), rsta.flatten())(xnew)
        rsta=np.flip(sta_smooth,axis=0)
        msta=np.mean(Mean_sta[c][len_sta:])
        
        from scipy.signal import chirp, find_peaks, peak_widths
        import matplotlib.pyplot as plt
        
        peaks, v_peaks = find_peaks(rsta)
        results_half = peak_widths(rsta, peaks, rel_height=0.5)
        results_full = peak_widths(rsta, peaks, rel_height=1)
        
        #Ppeaks, _ = find_peaks(rsta)
        #Npeaks, _ = find_peaks(-rsta)
        
        Ppeaks=np.argmax(rsta[upshift:])
        Npeaks=np.argmin(rsta[upshift:])

        D1=Ppeaks*binlenght#-shift*binlenght# remove 100 ms before zero
        D2=Npeaks*binlenght#-shift*binlenght# remove 100ms before zero
        thrp=msta+abs(rsta[Ppeaks+upshift]-msta)/2
        
        for iP1 in range(len(rsta)-Ppeaks):
            if rsta[Ppeaks+upshift+iP1]<thrp:
               break
        if Npeaks>0:   
            for iP2 in range(Ppeaks+upshift):
                if rsta[Ppeaks+upshift-iP2]<thrp:
                    break
        else:
            iP2=-1 
            
        w1=abs((Ppeaks+iP2/2)-(Ppeaks-iP1/2))*binlenght
        
        thrn=msta-abs(rsta[Npeaks+upshift]-msta)/2
        
        if Npeaks>0:
            for iN1 in range(Npeaks+upshift):
                if rsta[Npeaks+upshift-iN1]>thrn:
                   break   
        else:
            iN1=-1   
        
        for iN2 in range(len(rsta)-Npeaks):
            if rsta[Npeaks+upshift+iN2]>thrn:
                break 
        w2=abs((Npeaks+iN2/2)-(Npeaks-iN1/2))*binlenght
        
       
        fig, ax =plt.subplots()

        plt.plot(rsta)
        plt.hlines(msta-abs(rsta[Npeaks+upshift]-msta)/2, xmin = Npeaks+upshift-iN1, xmax = Npeaks+upshift+iN2,color="C1") # width of negative peak
        plt.hlines(msta+abs(rsta[Ppeaks+upshift]-msta)/2, xmin = Ppeaks+upshift-iP2, xmax = Ppeaks+upshift+iP1,color="C30")  # width of positive peak
        
        plt.vlines(Ppeaks+upshift, ymin =msta , ymax = rsta[Ppeaks+upshift],color="C5")
        plt.vlines(Npeaks+upshift, ymin = rsta[Npeaks+upshift], ymax = msta,color="C5")
        
        plt.hlines(msta, xmin = 0, xmax = len(rsta),linestyles='dashed')
        plt.vlines(upshift, ymin = msta, ymax = rsta[Ppeaks+upshift],linestyles='dashed')


        
        plt.text(Ppeaks+upshift+5, thrp, " %.0f   " % (w1), fontsize=11)
        plt.text(Npeaks+upshift+5, thrn, " %.0f   " % (w2), fontsize=11)
        
        plt.plot(Ppeaks+upshift, rsta[Ppeaks+upshift], "x")
        plt.plot(Npeaks+upshift, rsta[Npeaks+upshift], "x")
        plt.title(np.str(len(Mean_sta)-c))
        #labels = [item.get_text() for item in ax.get_xticklabels()]
        

        #t2=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        t1=np.arange(0, len(rsta), step=50)*binlenght
        t=(t1-upshift*binlenght)/1000
        #t2=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
#        t1=np.arange(shift, len(rsta), step=3)
#        t2=(t1-2)*binlenght/1000
        #plt.xlabel('time s')
        #labels = tt
        plt.xticks(np.arange(0, len(rsta), step=50),t)

        #ax.set_xticklabels(labels)
        #xticks(tt)
        plt.savefig('STA_width_latency'+figname+np.str(len(Mean_sta)-c)+'.pdf', bbox_inches='tight')       
       
       
     
            

#        D1w=  iN*40    # width of positive peak
#        D2w= ((Npeaks+iN)-(Ppeaks+iP))*40# width of negative peak
#        
#        plt.plot(rsta)
#        plt.hlines(msta, xmin = Ppeaks+iP, xmax = Npeaks+iN,color="C1") # width of negative peak
#        plt.hlines(msta, xmin = 0, xmax = Ppeaks+iP,color="C30")  # width of positive peak
#        
#        plt.vlines(Ppeaks, ymin =msta , ymax = rsta[Ppeaks],color="C5")
#        plt.vlines(Npeaks, ymin = rsta[Npeaks], ymax = msta,color="C5")
#        
#        plt.text(Ppeaks, msta, " %.0f   " % (D1w), fontsize=11)
#        plt.text(Npeaks, msta, " %.0f   " % (D2w), fontsize=11)
#        
#        plt.plot(Ppeaks, rsta[Ppeaks], "x")
#        plt.plot(Npeaks, rsta[Npeaks], "x")
#        plt.title(np.str(c))
#        plt.savefig('STA_width_latency '+np.str(c)+'.pdf', bbox_inches='tight')

        plt.show()
        tabel.append((len(Mean_sta)-c,w1,w2,D1,D2))
    np.save('latncy_widths_STA_'+figname,tabel)
   
    return tabel