"""
=========
plot_PSD
=========

My Entity model template The System Development Kit
Used as a template for all TheSyDeKick Entities.

Current docstring documentation style is Numpy
https://numpydoc.readthedocs.io/en/latest/format.html

This text here is to remind you that documentation is important.
However, youu may find it out the even the documentation of this 
entity may be outdated and incomplete. Regardless of that, every day 
and in every way we are getting better and better :).

Initially written by Marko Kosunen, marko.kosunen@aalto.fi, 2017.

"""

import os
import sys
if not (os.path.abspath('../../thesdk') in sys.path):
    sys.path.append(os.path.abspath('../../thesdk'))

from thesdk import *

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig


import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

def plot_PSD(**kwargs):
        """ Method for plotting PSD of one signal. If BW is given and no_plot is 0, will add ACLR levels to figure.

            Parameters
            ----------
            signal : array 
                Signal to be plotted
            Fc : integer
                Carrier frequency
            Fs : integer
                Sampling frequency
            BW : integer
                Bandwidth of whole signal
            no_plot : binary
                If 1, will return y and x- axis data (and ACLR) without plotting.

            Example
            -------
            self.plot_PSD(signal=[1+....-0.5],Fc=1e9,Fs=10e9, BW=200e6)

        """

        BW=0
        Fc=0
        no_plot=0
        a=100
        x=[] #self.rf_signal
        if 'BW' in kwargs:
            BW=kwargs.get('BW')
        if 'Fc' in kwargs:
            Fc=kwargs.get('Fc')
        if 'Fs' in kwargs:
            Fs=int(kwargs.get('Fs'))

        if 'no_plot' in kwargs:
            no_plot=kwargs.get('no_plot')
        if 'signal' in kwargs:
            x=kwargs.get('signal')
     

        #s=x.NRfilter(Fs,s,BW,x.self.osr)
        logging.info('Initializing')
        s=x[:,1]+1j*x[:,2]
        Lsegm_perc = 10
        Fs_given = 0
        plot_color = 'k'
        win_type = 'tukey'
        param = 1
        overlap_perc = 50

        Fs_given = 1
        
        fmin = -Fs/2
        fmax = Fs/2

        #Lsegm_perc = a
        #a=len(s)
        Lsegm = round(len(s)*Lsegm_perc/100)
        noverlap = round(Lsegm * overlap_perc/100)

        logging.info('Before win')

        win=sig.tukey(Lsegm,param)

        print('Lsegm',Lsegm)
        #print('length',len(Lsegm))
        print('win',win)
        print('len win',len(win))

        logging.info('after win')

        f,Pxx=sig.welch(np.real(s),Fs,win,Lsegm,noverlap=noverlap,detrend=False, return_onesided=True)
        #f,Pxx=sig.welch(np.real(s),Fs,noverlap=noverlap,detrend=False, return_onesided=True)

        logging.info('After wlsch func')

        
        y=10*np.log10(Pxx/max(Pxx))
        
        L = len(f)
        n1 = round((L-1)/Fs*fmin + (L+1)/2)
        n2 = round((L-1)/Fs * fmax + (L+1)/2)
        f_plot =f # f[n1-1:n2]
        y_plot =y # y[n1-1:n2]

        o=f_plot.argsort()
        logging.info('After wlsch func n after initialization')

        if no_plot==0:
            logging.info('I am here in no plot')
            fig,ax=plt.subplots()
            plt.plot(f_plot[o]/(10**6),y_plot[o])
            plt.grid(b=True)
            plt.title("Signal spectrum")
            logging.info('done with no plot')


        #pdb.set_trace()
        if BW!=0:
            logging.info('starting with BW')
            ACLR=[]
            matrix=np.transpose(np.vstack((f,Pxx)))
            matrix=matrix[matrix[:,0].argsort()]
            l=(np.abs(matrix[:,0]-(Fc-0.5*BW))).argmin()
            h=(np.abs(matrix[:,0]-(Fc+0.5*BW))).argmin()
            chpow=np.mean(matrix[l:h,1])

            ncarrier=(max(f)-min(f))/BW
            if ncarrier>=5 and max(f)>Fc+2.5*BW and min(f)<Fc-2.5*BW:
                ncarrier=5
            elif ncarrier>=3 and max(f)>=Fc+1.5*BW and min(f)<=Fc-1.5*BW:
                ncarrier=3
            else:
                ncarrier=0

            for i in range(-int(np.floor(ncarrier/2)),int(np.floor(ncarrier/2)+1)):
                l=(np.abs(matrix[:,0]-(Fc+(i-0.5)*BW))).argmin()
                h=(np.abs(matrix[:,0]-(Fc+(i+0.5)*BW))).argmin()
                ac=np.mean(matrix[l:h,1])/chpow
                ACLR.append(10*np.log10(ac))
                #ax.bar((self.Fc+i*BW)/(10**6),10*np.log10(ac),BW/(10**6))
                if no_plot==0:
                    ax.hlines(10*np.log10(ac),(Fc+(i-0.5)*BW)/(10**6),(Fc+(i+0.5)*BW)/(10**6),label=str(ac),colors='r',zorder=10)
                    ax.text(x=(Fc+(i-0.5)*BW)/(10**6),y=10*np.log10(ac)+3,s=str(round(10*np.log10(ac),2)),fontsize='small',color='r')
            logging.info('done with no BW')

        if no_plot==0:
            #plt.xlim((0,2*Fc/(10**6)))
            plt.xlabel("Frequenzy [MHz]")
            plt.ylabel("PSD [dB]")
            plt.show(block=False)
        if no_plot==0:
            if BW!=0:
                return fig, ACLR
            return fig
        else:
            if BW!=0:
                return f_plot[o]/(10**6),y_plot[o], ACLR
            else:
                return f_plot[o]/(10**6),y_plot[o]

