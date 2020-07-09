import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.image import imread 
from matplotlib.colors import rgb_to_hsv 
import colorsys
from copy import deepcopy 

def myPlot(DATA, var, block, collor, xlab, ylab, xlim, ylim, ax, title='', subtitle='', legpos='lower left'):
    # Discrete variables plots e.g. RT, correct, MT
    # var: string; e.g. "RT", 'correct"
    # block: 50 or 90

    DATA = DATA.astype({'block': 'int'})
    labelPlot = ['INCONG', 'CONG']
    x = DATA['ratioDist'].unique() / 1000  # I used ratios on the hundreds i.e. divide by 1000 to get actual ratios
    x.sort()
    y = DATA.groupby(['subjID', 'ratioDist', 'condition', 'block']).mean().reset_index()
    y_mean = y.groupby(['ratioDist', 'condition', 'block']).mean().reset_index()
    y_sem = y.groupby(['ratioDist', 'condition', 'block']).sem().reset_index()

    y1 = y_mean[['ratioDist', 'condition', 'block', var]]
    y1 = y1.loc[y1.loc[:, 'block'] == block, :]
    y1_sem = y_sem[['ratioDist', 'condition', 'block', var]]
    y1_sem = y1_sem.loc[y1_sem.loc[:, 'block'] == block, :]
    ax.plot(x, y1.loc[y1['condition'] == 'Congruent', var], 'o', color=collor)
    ax.errorbar(x, y1.loc[y1['condition'] == 'Congruent', var],
                y1_sem.loc[y1_sem['condition'] == 'Congruent', var],
                label=labelPlot[1], color=collor)

    ax.plot(x, y1.loc[y1['condition'] == 'Incongruent', var], 'o', color=collor)
    ax.errorbar(x, y1.loc[y1['condition'] == 'Incongruent', var],
                y1_sem.loc[y1_sem['condition'] == 'Incongruent', var],
                dashes=[2, 4], label=labelPlot[0], color=collor)

    if legpos != 'none':
        ax.plot(-10, -10, color=(255 / 255, 0 / 255, 0 / 255, 1), label='Block 90')
        ax.plot(-10, -10, color=(34 / 255, 139 / 255, 24 / 255, 1), label='Block 50')
        ax.legend(loc=legpos)

    ax.axis([xlim[0], xlim[1], ylim[0], ylim[1]])
    ax.set_xticks(x)
    ax.set_xlabel(xlab, fontsize=14)
    ax.set_ylabel(ylab, fontsize=14)
    ax.set_title(title, fontsize=13)


def my_filter(img, hue_range):
    # filters image by desired color
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # define range of color in HSV
    # In opencv hue range is [0,179], saturation range is [0,255], and value range is [0,255]
    lower_color = np.array([hue_range[0], 0, 0], dtype=np.float64)
    upper_color = np.array([hue_range[1], 255, 255], dtype=np.float64)
    # Threshold the HSV image to get only desired colors
    mask = cv2.inRange(hsv, lower_color, upper_color)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img, img, mask=mask)

    return mask, res


def format_axes(fig, axes_to_format):
    counter = 0
    for i, ax in enumerate(fig.axes):
        if i == axes_to_format[counter]:
            ax.axis('off')
            counter = counter + 1
        if counter == len(axes_to_format):
            break


def img_dist(numerator, denominator, domain):
    #numerator, denominator: np.array
    #domain: str
    
    idx1 = np.logical_and(~np.isnan(numerator), numerator != 0)
    idx2 = np.logical_and(~np.isnan(denominator), denominator != 0)
    idx = np.logical_and(idx1, idx2)
    n1 = numerator[idx]
    numerators = pd.DataFrame(n1)
    numerators = numerators.transpose()

    d1 = denominator[idx]
    denominators = pd.DataFrame(d1)
    denominators = denominators.transpose()

    distr = num_den_dist(numerators, denominators, domain, 1)

    return distr


def num_den_dist(numerators, denominators, domain, eq, tipo = 1):
    #Returns the frequency when the larger fraction has the larger numerator and/or denominator.
    #It allows for self-comparison (e.g. the same image compared to itself e.g. ne_de cases)
    #Inputs:
    #numerators: dataframe(2 x n), each row is for a ratio, and n the number of ratio pairs
    #denominators: dataframe(2 x n)
    #domain of data: vector with strings [domain, num. name, den. name, category]
    #eq: 1 if the ratio pairs are identical in numerators/denominators; 0 if not.
    #tipo 1 RL, 2 RS, 3 RE ...

    n1 = numerators.iloc[0,:].values
    d1 = denominators.iloc[0,:].values
   

    r1 = n1 / d1
    n_ratio_pairs = len(r1)
    ns_dl_rl = 0  # larger ratio has smaller numerator and larger denominator (not possible, should be zero!)
    nl_dl_rl = 0  # larger ratio has larger numerator and denominator
    ns_ds_rl = 0  # larger ratio has smaller numerator and denominator
    nl_ds_rl = 0  # larger ratio has larger numerator and smaller denominator
    ne_dl_rl = 0  # larger ratio has equal numerator and larger denominator (not possible, should be zero!)
    ne_ds_rl = 0  # larger ratio has equal numerator and smaller denominator
    ns_de_rl = 0  # larger ratio has smaller numerator and equal denominator (not possible, should be zero!)
    nl_de_rl = 0  # larger ratio has larger numerator and equal denominator
    ne_de_rl = 0  # larger ratio has equal numerator and equal denominator (not applicable, the counts reflect how many)

    ns_dl = 0
    nl_dl = 0
    ns_ds = 0
    nl_ds = 0
    ne_dl = 0
    ne_ds = 0
    ns_de = 0
    nl_de = 0
    ne_de = 0

    rep = np.repeat
    logic_and = np.logical_and
    for rp in range(n_ratio_pairs):
        #print(rp)
        rA = rep(r1[rp], n_ratio_pairs)
        nA = rep(n1[rp], n_ratio_pairs)
        dA = rep(d1[rp], n_ratio_pairs)

     
        ns_dl += sum(logic_and(nA < n1, dA > d1))
        nl_dl += sum(logic_and(nA > n1, dA > d1))
        ns_ds += sum(logic_and(nA < n1, dA < d1))
        nl_ds += sum(logic_and(nA > n1, dA < d1))
        ne_dl += sum(logic_and(nA == n1, dA > d1))
        ne_ds += sum(logic_and(nA == n1, dA < d1))
        ns_de += sum(logic_and(nA < n1, dA == d1))
        nl_de += sum(logic_and(nA > n1, dA == d1))
        ne_de += sum(logic_and(nA == n1, dA == d1)) 


        idx1 = rA > r1
        if tipo == 2:
            idx1 = rA < r1
        elif tipo == 3:
            idx1 = rA == r1


        n1_L = nA[idx1]
        d1_L = dA[idx1]
        n1_S = n1[idx1]
        d1_S = d1[idx1]

        ns_dl_rl += sum(logic_and(n1_L < n1_S, d1_L > d1_S))
        nl_dl_rl += sum(logic_and(n1_L > n1_S, d1_L > d1_S))
        ns_ds_rl += sum(logic_and(n1_L < n1_S, d1_L < d1_S))
        nl_ds_rl += sum(logic_and(n1_L > n1_S, d1_L < d1_S))
        ne_dl_rl += sum(logic_and(n1_L == n1_S, d1_L > d1_S))
        ne_ds_rl += sum(logic_and(n1_L == n1_S, d1_L < d1_S))
        ns_de_rl += sum(logic_and(n1_L < n1_S, d1_L == d1_S))
        nl_de_rl += sum(logic_and(n1_L > n1_S, d1_L == d1_S))
        ne_de_rl += sum(logic_and(n1_L == n1_S, d1_L == d1_S))

        
    distr = pd.DataFrame([ns_dl_rl, nl_dl_rl, ns_ds_rl, nl_ds_rl,
                          ne_dl_rl, ne_ds_rl, ns_de_rl, nl_de_rl,
                          ne_de_rl])
    distr_total = pd.DataFrame([ns_dl, nl_dl, ns_ds, nl_ds,
                          ne_dl, ne_ds, ns_de, nl_de,
                          ne_de])
    cat = pd.DataFrame(['NS_DL','NL_DL','NS_DS','NL_DS',
                        'NE_DL','NE_DS','NS_DE','NL_DE',
                        'NE_DE'])
    #do1 = pd.DataFrame(rep(domain[0], cat.shape[0]))
    #do2 = pd.DataFrame(rep(domain[1], cat.shape[0]))
    #do3 = pd.DataFrame(rep(domain[2], cat.shape[0]))
    #do4 = pd.DataFrame(rep(domain[3], cat.shape[0]))
    #distr = [cat, distr, do1, do2, do3, do4, distr_total]
    distr = [cat, distr, distr_total]
    distr = pd.concat(distr, axis=1)
    distr.columns = ['Type','Frequency', 'Total']
    #distr.columns = ['Type','Frequency','Domain','Num','Den','Category', "Total"]
    return distr

def img_counts(dirr, name_files, progress = False):
    NUM = [[], [], [], []] #pixels of hue
    DEN = [[], [], [], []]
    NUM_B = [[], [], [], []] #brightness of hue
    DEN_B = [[], [], [], []]
    NUM_S = [[], [], [], []]  #saturation of hue
    DEN_S = [[], [], [], []]
    for idx, img in enumerate(name_files):
        if progress: #completion (%)
            print(str(np.round((idx+1)/len(name_files), 2)) + " " + img)
        
        BGRp = cv2.imread(dirr + img) 
        RGBp = cv2.cvtColor(BGRp, cv2.COLOR_BGR2RGB) # RGB values for each image of the artist/domain as an np.array
        HSVp = cv2.cvtColor(RGBp, cv2.COLOR_RGB2HSV)
       
        

        img_dim = RGBp.shape  # [height, width]
        npixels = img_dim[0] * img_dim[1]

        k = 179/360 # open cv hue range 0,179;
        hue_ranges = [[0, 60*k], #red-yellowish
                      [60*k, 160*k], #yellow-greenish
                      [160*k, 250*k], #green-blueish
                      [250*k, 360*k]] #blueish-purple

        for idx2, hue in enumerate(hue_ranges):
            mask1, res1 = my_filter(RGBp, hue)
            hsv = cv2.cvtColor(res1, cv2.COLOR_RGB2HSV)


            idx3 = mask1!=0
            #hue
            NUM[idx2].append(idx3.sum())
            DEN[idx2].append(npixels)

            #saturation
            NUM_S[idx2].append(hsv[idx3,1].sum())
            DEN_S[idx2].append(HSVp[:,:,1].sum())

            #brightness (value)
            NUM_B[idx2].append(hsv[idx3,2].sum())
            DEN_B[idx2].append(HSVp[:,:,2].sum())


    return [NUM, DEN, NUM_B, DEN_B, NUM_S, DEN_S]

def my_total(distr):
    # Total number of comparisons regardless if the ratio is RL or RS 
    # It's a sum because being the larger or smaller ratio is relative 
    # e.g. if a ratio is larger and NLDL, the same ratio is an NSDS when looking for smaller ratios  
    # This function is ONLY for the education material with explicit pairs of fractions in the exercise
    # distr: vector with cases, ordered as: 'NS_DL','NL_DL','NS_DS','NL_DS', 'NE_DL', 'NE_DS', 'NS_DE','NL_DE', 'NE_DE'
    
    distr_all = []
    distr_all.append(distr[0] + distr[3]) #NS_DL (NS_DL is NL_DS when its the small ratio)
    distr_all.append(distr[1] + distr[2]) #NL_DL (NL_DL is NS_DS when its the small ratio)
    distr_all.append(distr[1] + distr[2]) #NS_DS (NS_DS is NL_DL when its the small ratio)
    distr_all.append(distr[0] + distr[3]) #NL_DS (NL_DS is NS_DL when its the small ratio)
    distr_all.append(distr[4] + distr[5]) #NE_DL (NE_DL is NE_DS when its the small ratio)
    distr_all.append(distr[4] + distr[5]) #NE_DS (NE_DS is NE_DL when its the small ratio)
    distr_all.append(distr[6] + distr[7]) #NS_DE (NS_DE is NL_DE when its the small ratio)
    distr_all.append(distr[6] + distr[7]) #NL_DE (NL_DE is NS_DE when its the small ratio)
    distr_all.append(0) #NE_DE (ratio is the same)
    
    return distr_all

def my_posterior(typpe, distr, distr_all):
    #type NL, NS, DL, DS; str i.e. to compute p(RL|type)
    #distr: vector of p(cases|RL) i.e. c(rl_nsdl, rl_nldl, rl_nsds, rl_nlds, rl_nedl, rl_neds, rl_nsde, rl_nlde, rl_nede)
    #distr_all: vector of cases p(cases) i.e. c(nsdl, nldl, nsds, nlds, nedl, neds, nsde, nlde, nede)
  
    if typpe == 'NL':
        type_rl = (distr[1]+distr[3]+distr[7])/sum(distr) #p(NL|RL)
        rl = sum(distr)/sum(distr_all) #p(RL)
        typpe = (distr_all[1]+distr_all[3]+distr_all[7])/sum(distr_all) #p(NL)
        posterior = (type_rl*rl)/(typpe) #posterior p(RL|NL)
    elif typpe == 'NS':
        type_rl = (distr[0]+distr[2]+distr[6])/sum(distr) #p(NS|RL)
        rl = sum(distr)/sum(distr_all) #p(RL)
        typpe = (distr_all[0]+distr_all[2]+distr_all[6])/sum(distr_all) #p(NS)
        posterior = (type_rl*rl)/(typpe) #posterior p(RL|NS)
    elif typpe == 'DL':
        type_rl = (distr[0]+distr[1]+distr[4])/sum(distr) #p(DL|RL)
        rl = sum(distr)/sum(distr_all) #p(RL)
        typpe = (distr_all[0]+distr_all[1]+distr_all[4])/sum(distr_all) #p(DL)
        posterior = (type_rl*rl)/(typpe) #posterior p(RL|DL)
    elif typpe == 'DS':
        type_rl = (distr[2]+distr[3]+distr[5])/sum(distr) #p(DS|RL)
        rl = sum(distr)/sum(distr_all) #p(RL)
        typpe = (distr_all[2]+distr_all[3]+distr_all[5])/sum(distr_all) #p(DS)
        posterior = (type_rl*rl)/(typpe) #posterior p(RL|DS)
         
    return posterior
 
 
def dropdown_callback(hue, hsv_dim, img1, img2, dirr, NUMS, DENS):
    
    NUM = NUMS[0]
    NUM_B = NUMS[1]
    NUM_S = NUMS[2]
    
    DEN = DENS[0]
    DEN_B = DENS[1]
    DEN_S = DENS[2]
    
    # Counts for selected images
    results_s = img_counts(dirr, [img1, img2], progress = False) 

    #load img1 and img2 
    img1 = cv2.cvtColor(cv2.imread(dirr + img1), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(dirr + img2), cv2.COLOR_BGR2RGB)
    
    
   
    #Hue information
    k = 179/360 # open cv hue 0,179; s,v 0,255; but in the paper we used a 360 reference 
    #hue = 3 #0 to 3; we divided the hue feature in 4 bands (See paper)
    hue_ranges = [[0, 60*k], #red-yellowish
                  [60*k, 160*k], #yellow-greenish
                  [160*k, 250*k], #green-blueish
                  [250*k, 360*k]] #blueish-purple See paper
    #Posteriors
    #hsv_dim = 'hue'
    if hsv_dim == 'hue':
        numerator = np.array(NUM[hue])
        denominator = np.array(DEN[hue])
        NUM_s = np.array(results_s[0][hue])/1000 #selected images
        DEN_s = np.array(results_s[1][hue])/1000
    elif hsv_dim == 'brightness':
        numerator = np.array(NUM_B[hue])
        denominator = np.array(DEN_B[hue])
        NUM_s = np.array(results_s[2][hue])/1000 #selected images
        DEN_s = np.array(results_s[3][hue])/1000
    elif hsv_dim == 'saturation':
        numerator = np.array(NUM_S[hue])
        denominator = np.array(DEN_S[hue])
        NUM_s = np.array(results_s[4][hue])/1000 #selected images
        DEN_s = np.array(results_s[5][hue])/1000


    distr = img_dist(numerator, denominator, domain = dirr)

    RL_NL = my_posterior('NL', distr['Frequency'], distr['Total'])
    RL_NS = my_posterior('NS', distr['Frequency'], distr['Total'])
    RL_DL = my_posterior('DL', distr['Frequency'], distr['Total'])
    RL_DS = my_posterior('DS', distr['Frequency'], distr['Total'])

    #print([RL_NL,RL_NS,RL_DL,RL_DS])

    mask1, res1 = my_filter(img1, hue_ranges[hue])
    mask2, res2 = my_filter(img2, hue_ranges[hue])
    
    fig = plt.figure(constrained_layout=True, figsize = (10,7.5))
    gs = GridSpec(3, 8, figure = fig)
    ax1 = fig.add_subplot(gs[0, 1:4])
    ax2 = fig.add_subplot(gs[0, 4:7])
    ax3 = fig.add_subplot(gs[1, 1:4])
    ax4 = fig.add_subplot(gs[1, 4:7])
    ax5 = fig.add_subplot(gs[0, 0])
    ax6 = fig.add_subplot(gs[1, 0])
    ax7 = fig.add_subplot(gs[2, 1:4])
    ax8 = fig.add_subplot(gs[2, 4:7])
    ax1.imshow(img1)
    ax2.imshow(img2)
    ax3.imshow(res1)
    ax3.set_title("Num: " + str(round(NUM_s[0],1)) + " Den: " + str(round(DEN_s[0],1)) + " Ratio: " + str(round(NUM_s[0]/DEN_s[0],1)))
    ax4.imshow(res2)
    ax4.set_title("Num: " + str(round(NUM_s[1],1)) + " Den: " + str(round(DEN_s[1],1)) + " Ratio: " + str(round(NUM_s[1]/DEN_s[1],2)))
    ax5.text(0.5, 0.5, "Original", va = "center", ha = "center", 
             rotation=90, fontsize = 17)
    ax6.text(0.5, 0.5, "Filtered", va = "center", ha = "center", 
             rotation=90, fontsize = 17)
    ax7.scatter(x=numerator/denominator, y=denominator/1000)
    ax7.set_xlabel('Ratio', fontsize=12)
    ax7.set_ylabel('Denominator (a.u.)', fontsize=12)
    ax7.set_title('Ratio-denominator relationship \n (dot: image in the category)')
    #ax7.set_ylim([0,1000000])
    ax8.bar(np.arange(4), np.array([RL_NL, RL_NS, RL_DL, RL_DS]))#,
           #tick_label = ('Num. Larger (NL)','Num. Smaller (NS)','Den. Larger (DL)','Den. Smaller (DS)'))
    #ax8.set_xticklabels(labels = ['Num. Larger (NL)','Num. Smaller (NS)','Den. Larger (DL)','Den. Smaller (DS)'],
    #                    fontdict = {'rotation': 90, 'ha': "center", 'fontsize': 10})
    plt.xticks(np.arange(4),['Num. Larger (NL)','Num. Smaller (NS)','Den. Larger (DL)','Den. Smaller (DS)'],
              rotation=45, ha="right")
    ax8.set_ylim([0,1])
    ax8.set_ylabel('p(RL | ...)', fontsize=12)
    ax8.set_title('Posteriors for larger ratio (RL) \n (all paired comparisons in the category)')
    fig.suptitle("Example of a paired comparison and summary information \n HSV dimension: " + hsv_dim , fontsize = 15)
    format_axes(fig, [0,1,2,3,4,5])
    fig.savefig('examples_posteriors.pdf')
    

