''' 
calculate RSA similarity matrix for each layer
'''
## Import necessary libraries
import scipy.io 
from scipy import stats
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt


layers = ['block1','block2','block3','block4','output']
## For each layer
for l in layers:
    ## Load the RDM (500x500) for each trained DNN
    autenc = scipy.io.loadmat('./tools/RDMs/autoencoder_encoder_'+l)['rdm']
    rgb2depth = scipy.io.loadmat('./tools/RDMs/rgb2depth_encoder_'+l)['rdm']
    keypoint2d = scipy.io.loadmat('./tools/RDMs/keypoint2d_encoder_'+l)['rdm']

    auto_arr = []
    keypoint2d_arr = []
    rgb2depth_arr = []

    count = 0
    ## Store the elements of the lower triangle of RDM.
    for i,j,k in zip(autenc,rgb2depth,keypoint2d):
        auto_arr.extend(i[:count])
        keypoint2d_arr.extend(k[:count])
        rgb2depth_arr.extend(j[:count])
        count+=1
    
    labels = ['autoencoders','kypoint2d','rgb2depth']
    similarity = np.zeros((3,3))
    sim_dict = {0:auto_arr,1:keypoint2d_arr,2:rgb2depth_arr}

    ## Perform RSA
    for val in sim_dict:
        similarity[val][val] = 1
    similarity[0][1] = similarity[1][0] = stats.spearmanr(sim_dict[0],sim_dict[1])[0]
    similarity[0][2] = similarity[2][0] = stats.spearmanr(sim_dict[0],sim_dict[2])[0]
    similarity[1][2] = similarity[2][1] = stats.spearmanr(sim_dict[1],sim_dict[2])[0]

    ax = sns.heatmap(similarity,xticklabels=labels,yticklabels=labels)
    fig = ax.get_figure()
    ## Save the output figure
    fig.savefig(l[:-3]+".png")