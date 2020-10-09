import numpy as np
import matplotlib.pyplot as plt

def plot_well_feature(x_trainwell, test_loc, test_inc, d,tvd):
    
    interval1 = [test_loc[0]*d+ tvd,(test_loc[0]+test_inc)*d+ tvd]

    interval2 = [test_loc[1]*d+ tvd,(test_loc[1]+test_inc)*d+ tvd] 
    
    
    depth = d * np.linspace(0.0, len(x_trainwell), num = len(x_trainwell)) + tvd
    
    fig = plt.figure(figsize=(12, 8))
    fig.set_facecolor('white')
    ax1 = fig.add_axes([0.07, 0.1, 0.1, 0.8],
                       ylim=(depth[0] ,depth[-1]))
    ax2 = fig.add_axes([0.18, 0.1, 0.1, 0.8],
                       ylim=(depth[0] ,depth[-1]))
    ax3 = fig.add_axes([0.29, 0.1, 0.1, 0.8],
                       ylim=(depth[0] ,depth[-1]))
    ax4 = fig.add_axes([0.40, 0.1, 0.1, 0.8],
                       ylim=(depth[0] ,depth[-1]))
    ax5 = fig.add_axes([0.51, 0.1, 0.1, 0.8],
                       ylim=(depth[0] ,depth[-1]))
    # ax6 = fig.add_axes([0.61, 0.1, 0.1, 0.8],
    #                    ylim=(depth[0] ,depth[-1]))
    # ax7 = fig.add_axes([0.72, 0.1, 0.1, 0.8],
    #                    ylim=(depth[0] ,depth[-1]))
    
    # ax1.plot(x_trainwell[:,0],depth,'-k', lw = 2)
    # ax1.fill_between([np.min(x_trainwell[:,0])-0.1*np.min(x_trainwell[:,0]),np.max(x_trainwell[:,0])+0.1*np.min(x_trainwell[:,0])],[interval1[0]],[interval1[1]],color="red",alpha=0.7)
    # ax1.fill_between([np.min(x_trainwell[:,0])-0.1*np.min(x_trainwell[:,0]),np.max(x_trainwell[:,0])+0.1*np.min(x_trainwell[:,0])],[interval2[0]],[interval2[1]],color="red",alpha=0.7)
    # ax1.set_xlim([np.min(x_trainwell[:,0])-0.1*np.min(x_trainwell[:,0]),np.max(x_trainwell[:,0])+0.1*np.min(x_trainwell[:,0])])
    # ax1.invert_yaxis()
    # ax1.set_xlabel('CAL',fontsize= 14)
    # ax1.set_ylabel('Depth (m)',fontsize = 14)
    # ax1.grid(linestyle='-.',linewidth=1.5)
    # ax1.tick_params(labelsize = 12)  
    # ax1.spines['bottom'].set_linewidth(1.5)
    # ax1.spines['left'].set_linewidth(1.5)
    # ax1.spines['right'].set_linewidth(1.5)
    # ax1.spines['top'].set_linewidth(1.5)
    
    ax1.plot(x_trainwell[:,0],depth,'-k', lw = 2)
    ax1.fill_between([np.min(x_trainwell[:,0])-0.1*np.min(x_trainwell[:,0]),np.max(x_trainwell[:,0])+0.1*np.min(x_trainwell[:,0])],[interval1[0]],[interval1[1]],color="red",alpha=0.7)
    ax1.fill_between([np.min(x_trainwell[:,0])-0.1*np.min(x_trainwell[:,0]),np.max(x_trainwell[:,0])+0.1*np.min(x_trainwell[:,0])],[interval2[0]],[interval2[1]],color="red",alpha=0.7)
    ax1.set_xlim([np.min(x_trainwell[:,0])-0.1*np.min(x_trainwell[:,0]),np.max(x_trainwell[:,0])+0.1*np.min(x_trainwell[:,0])])
    ax1.invert_yaxis()
    ax1.set_xlabel('PHI',fontsize= 16)
    ax1.grid(linestyle='-.',linewidth=1.5)
    ax1.tick_params(labelsize = 14)  
    ax1.spines['bottom'].set_linewidth(1.5)
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['right'].set_linewidth(1.5)
    ax1.spines['top'].set_linewidth(1.5)
    ax1.set_ylabel('Depth (m)',fontsize = 16)
    # ax1.set_yticklabels([])
    
    ax2.plot(x_trainwell[:,1],depth,'-k', lw = 2)
    ax2.fill_between([np.min(x_trainwell[:,1])-0.1*np.min(x_trainwell[:,1]),np.max(x_trainwell[:,1])+0.1*np.min(x_trainwell[:,1])],[interval1[0]],[interval1[1]],color="red",alpha=0.7)
    ax2.fill_between([np.min(x_trainwell[:,1])-0.1*np.min(x_trainwell[:,1]),np.max(x_trainwell[:,1])+0.1*np.min(x_trainwell[:,1])],[interval2[0]],[interval2[1]],color="red",alpha=0.7)
    ax2.set_xlim([np.min(x_trainwell[:,1])-0.1*np.min(x_trainwell[:,1]),np.max(x_trainwell[:,1])+0.1*np.min(x_trainwell[:,1])])
    ax2.invert_yaxis()
    ax2.set_xlabel('GR',fontsize= 16)
    ax2.grid(linestyle='-.',linewidth=1.5)
    ax2.tick_params(labelsize = 14)  
    ax2.spines['bottom'].set_linewidth(1.5)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.spines['right'].set_linewidth(1.5)
    ax2.spines['top'].set_linewidth(1.5)
    ax2.set_yticklabels([])
    
    ax3.plot(x_trainwell[:,2],depth,'-k', lw = 2)
    ax3.fill_between([np.min(x_trainwell[:,2])-0.1*np.min(x_trainwell[:,2]),np.max(x_trainwell[:,2])+0.1*np.min(x_trainwell[:,2])],[interval1[0]],[interval1[1]],color="red",alpha=0.7)
    ax3.fill_between([np.min(x_trainwell[:,2])-0.1*np.min(x_trainwell[:,2]),np.max(x_trainwell[:,2])+0.1*np.min(x_trainwell[:,2])],[interval2[0]],[interval2[1]],color="red",alpha=0.7)
    ax3.set_xlim([np.min(x_trainwell[:,2])-0.1*np.min(x_trainwell[:,2]),np.max(x_trainwell[:,2])+0.1*np.min(x_trainwell[:,2])])
    ax3.invert_yaxis()
    ax3.set_xlabel('DR',fontsize= 16)
    ax3.grid(linestyle='-.',linewidth=1.5)
    ax3.tick_params(labelsize = 14)  
    ax3.spines['bottom'].set_linewidth(1.5)
    ax3.spines['left'].set_linewidth(1.5)
    ax3.spines['right'].set_linewidth(1.5)
    ax3.spines['top'].set_linewidth(1.5)
    ax3.set_yticklabels([])
    
    # ax5.plot(x_trainwell[:,4],depth,'-k', lw = 2)
    # ax5.fill_between([np.min(x_trainwell[:,4])-0.1*np.min(x_trainwell[:,4]),np.max(x_trainwell[:,4])+0.1*np.min(x_trainwell[:,4])],[interval1[0]],[interval1[1]],color="red",alpha=0.7)
    # ax5.fill_between([np.min(x_trainwell[:,4])-0.1*np.min(x_trainwell[:,4]),np.max(x_trainwell[:,4])+0.1*np.min(x_trainwell[:,4])],[interval2[0]],[interval2[1]],color="red",alpha=0.7)
    # ax5.set_xlim([np.min(x_trainwell[:,4])-0.1*np.min(x_trainwell[:,4]),np.max(x_trainwell[:,4])+0.1*np.min(x_trainwell[:,4])])
    # ax5.invert_yaxis()
    # ax5.set_xlabel('MR',fontsize= 14)
    # ax5.grid(linestyle='-.',linewidth=1.5)
    # ax5.tick_params(labelsize = 12)  
    # ax5.spines['bottom'].set_linewidth(1.5)
    # ax5.spines['left'].set_linewidth(1.5)
    # ax5.spines['right'].set_linewidth(1.5)
    # ax5.spines['top'].set_linewidth(1.5)
    # ax5.set_yticklabels([])
    
    ax4.plot(x_trainwell[:,3],depth,'-k', lw = 2)
    ax4.fill_between([np.min(x_trainwell[:,3])-0.1*np.min(x_trainwell[:,3]),np.max(x_trainwell[:,3])+0.1*np.min(x_trainwell[:,3])],[interval1[0]],[interval1[1]],color="red",alpha=0.7)
    ax4.fill_between([np.min(x_trainwell[:,3])-0.1*np.min(x_trainwell[:,3]),np.max(x_trainwell[:,3])+0.1*np.min(x_trainwell[:,3])],[interval2[0]],[interval2[1]],color="red",alpha=0.7)
    ax4.set_xlim([np.min(x_trainwell[:,3])-0.1*np.min(x_trainwell[:,3]),np.max(x_trainwell[:,3])+0.1*np.min(x_trainwell[:,3])])
    ax4.invert_yaxis()
    ax4.set_xlabel('PE',fontsize= 16)
    ax4.grid(linestyle='-.',linewidth=1.5)
    ax4.tick_params(labelsize = 14)  
    ax4.spines['bottom'].set_linewidth(1.5)
    ax4.spines['left'].set_linewidth(1.5)
    ax4.spines['right'].set_linewidth(1.5)
    ax4.spines['top'].set_linewidth(1.5)
    ax4.set_yticklabels([])
    
    ax5.plot(x_trainwell[:,4],depth,'-k', lw = 2)
    ax5.fill_between([np.min(x_trainwell[:,4])-0.01*np.min(x_trainwell[:,4]),np.max(x_trainwell[:,4])+0.01*np.min(x_trainwell[:,4])],[interval1[0]],[interval1[1]],color="red",alpha=0.7)
    ax5.fill_between([np.min(x_trainwell[:,4])-0.01*np.min(x_trainwell[:,4]),np.max(x_trainwell[:,4])+0.01*np.min(x_trainwell[:,4])],[interval2[0]],[interval2[1]],color="red",alpha=0.7,label="Testing")
    ax5.set_xlim([np.min(x_trainwell[:,4])-0.01*np.min(x_trainwell[:,4]),np.max(x_trainwell[:,4])+0.01*np.min(x_trainwell[:,4])])
    ax5.invert_yaxis()
    ax5.set_xlabel('RHO',fontsize= 16)
    ax5.grid(linestyle='-.',linewidth=1.5)
    ax5.tick_params(labelsize = 14)  
    ax5.spines['bottom'].set_linewidth(1.5)
    ax5.spines['left'].set_linewidth(1.5)
    ax5.spines['right'].set_linewidth(1.5)
    ax5.spines['top'].set_linewidth(1.5)
    ax5.set_yticklabels([])
    # ax7.legend(bbox_to_anchor=(0.9, 1))
    
def plot_well_target(y_trainwell, test_loc, test_inc, d, tvd):
    
    interval1 = [test_loc[0]*d+ tvd,(test_loc[0]+test_inc)*d+ tvd]

    interval2 = [test_loc[1]*d+ tvd,(test_loc[1]+test_inc)*d+ tvd] 
    
    
    depth = d * np.linspace(0.0, len(y_trainwell), num = len(y_trainwell)) + tvd
    

    fig = plt.figure(figsize=(4, 8))
    fig.set_facecolor('white')
    ax1 = fig.add_axes([0.2, 0.1, 0.31, 0.8],
                    ylim=(depth[0] ,depth[-1]))
    
    ax1.plot(y_trainwell,depth,'-k', lw = 2)
    ax1.fill_between([np.min(y_trainwell)-0.1*np.min(y_trainwell),np.max(y_trainwell)+0.1*np.min(y_trainwell)],[interval1[0]],[interval1[1]],color="red",alpha=0.7)
    ax1.fill_between([np.min(y_trainwell)-0.1*np.min(y_trainwell),np.max(y_trainwell)+0.1*np.min(y_trainwell)],[interval2[0]],[interval2[1]],color="red",alpha=0.7,label="Testing")
    ax1.set_xlim([np.min(y_trainwell)-0.1*np.min(y_trainwell),np.max(y_trainwell)+0.1*np.min(y_trainwell)])
    ax1.invert_yaxis()
    ax1.set_xlabel('DTC',fontsize= 16)
    ax1.set_ylabel('Depth (m)',fontsize = 16)
    ax1.grid(linestyle='-.',linewidth=1.5)
    ax1.tick_params(labelsize = 14)  
    ax1.spines['bottom'].set_linewidth(1.5)
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['right'].set_linewidth(1.5)
    ax1.spines['top'].set_linewidth(1.5)
    

def plot_well_predict(Y_train, Y_predict, test_loc, test_inc, d,tvd):
    
    for i in range(len(test_loc)):
    
        depth = d * np.linspace(0.0, test_inc, num = int(test_inc)) + test_loc[i] * d + tvd
        
        
        fig = plt.figure(figsize=(9, 4))
        ax = fig.add_subplot()
        fig.set_facecolor('white')
        plt.plot(depth,Y_train[(i)*test_inc:(i+1)*test_inc],'r',label="Reference",lw = 2)
        
        plt.plot(depth,Y_predict[0,(i)*test_inc:(i+1)*test_inc],'g',label="NN",lw = 2)
        
        plt.plot(depth,Y_predict[1,(i)*test_inc:(i+1)*test_inc],'b',label="BNN_1st",lw = 2)
        
        plt.plot(depth,Y_predict[2,(i)*test_inc:(i+1)*test_inc],'b--',label="BNN_2nd",lw = 2)
        
        plt.plot(depth,Y_predict[3,(i)*test_inc:(i+1)*test_inc],'b:',label="BNN_3rd",lw = 2)
        
        if i == 0:
            plt.legend(loc='best',fontsize= 12)  #112,140  108,180
        else:
            plt.legend(loc='best',fontsize= 12)  #111, 150 110,190
        
        
        # plt.legend(loc='best')
        ax.set_xlabel('Depth (m)',fontsize= 16)
        ax.set_ylabel('DTC (ns/ft)',fontsize = 16)
        
        if i == 0:
            ax.set_ylim([60,80.00001])  #112,140  108,180
        else:
            ax.set_ylim([65,90])  #111, 150 110,190
        
        ax.grid(linestyle='-.',linewidth=1.5)
        ax.tick_params(labelsize = 14)  
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
        
        # if i == 0:
        
        #     plt.savefig('/Users/Runhai/Documents/TUD/Paper/BNN/Fig/Fig_1.png',dpi=500, facecolor='w', edgecolor='w',bbox_inches='tight', 
        #           transparent=True)
            
        # else:
            
        #     plt.savefig('/Users/Runhai/Documents/TUD/Paper/BNN/Fig/Fig_2.png',dpi=500, facecolor='w', edgecolor='w',bbox_inches='tight', 
        #           transparent=True)
            
    
    
def plot_pred_interval(S, Y_test, Y_predict, test_loc, test_inc, d, tvd):    
    
    
    for i in range(len(test_loc)):
    
        depth = d * np.linspace(0.0, test_inc, num = int(test_inc)) + test_loc[i] * d + tvd
        
        sigma = S[(i)*test_inc:(i+1)*test_inc]

        Y = Y_predict[(i)*test_inc:(i+1)*test_inc]
        
        fig = plt.figure(figsize=(9, 4))
        ax = fig.add_subplot()
        fig.set_facecolor('white')
        plt.plot(depth,Y_test[(i)*test_inc:(i+1)*test_inc],'r',label="Reference",lw=2)
        plt.plot(depth,Y,'b',label="Ensemble Mean",lw=2)
        plt.fill_between(depth,Y + sigma, Y - sigma,color='b', alpha=.3,label="+/- $\sigma$")
        
        plt.fill_between(depth,Y + 2*sigma, Y - 2*sigma,color='b', alpha=.2,label="+/- 2$\sigma$")
        
        plt.fill_between(depth,Y + 4*sigma, Y - 4*sigma,color='b', alpha=.1,label="+/- 4$\sigma$")
        
        if i == 0:
            plt.legend(loc='best',fontsize= 12)  #112,140  108,180
        else:
            plt.legend(loc='best',fontsize= 12)  #111, 150 110,190
        
        
        # plt.legend(loc='best')
        ax.set_xlabel('Depth (m)',fontsize= 16)
        ax.set_ylabel('DTC (μs/ft)',fontsize = 16)
        
        if i == 0:
            ax.set_ylim([60,80.00001])  #112,140  108,180
        else:
            ax.set_ylim([65,90])  #111, 150 110,190
        
        ax.grid(linestyle='-.',linewidth=1.5)
        ax.tick_params(labelsize = 14)  
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
        
        # if i == 0:
        
        #     plt.savefig('/Users/Runhai/Documents/TUD/Paper/BNN/Fig/Fig_3_1.png',dpi=500, facecolor='w', edgecolor='w',bbox_inches='tight', 
        #           transparent=True)
            
        # else:
            
        #     plt.savefig('/Users/Runhai/Documents/TUD/Paper/BNN/Fig/Fig_4_1.png',dpi=500, facecolor='w', edgecolor='w',bbox_inches='tight', 
        #           transparent=True)
            

    
