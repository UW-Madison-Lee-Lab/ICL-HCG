import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pickle
import torch
import numpy as np
import os
import sys
sys.path.insert(0, '..')
from utils import to_python_float
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Font sizes
FS = 28
x_label_fontsize = FS + 8
y_label_fontsize = FS + 8
title_fontsize   = FS + 8
legend_fontsize  = FS + 8
tick_fontsize    = FS

# Custom line width
LW = 6

# runs
random_seeds   = [2023, 2024, 2025, 2026]

models = {
    'transformer': 'Transformer',
    'mamba': 'Mamba',
    'lstm': 'LSTM',
    'gru': 'GRU',
    }

setup = 'IOHypothesis+Size_h+xy+z'
step_name      = 'global_step'
x_label_name   = 'Epoch'
folder_name = f'FIG_{setup}'

if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    
random_seed_list = [2023,2024,2025,2026]
table_length_list = [
    14,13,12,11,10,
     9, 8, 7,
     6, 5, 4, 3, 2,
]
title_sub_list = [
    r'$\boldsymbol{OOD}$', r'$\boldsymbol{OOD}$', r'$\boldsymbol{OOD}$', r'$\boldsymbol{OOD}$', r'$\boldsymbol{OOD}$',
    r'$\boldsymbol{ID}$' , r'$\boldsymbol{ID}$' , r'$\boldsymbol{ID}$' ,
    r'$\boldsymbol{OOD}$', r'$\boldsymbol{OOD}$', r'$\boldsymbol{OOD}$', r'$\boldsymbol{OOD}$', r'$\boldsymbol{OOD}$',
    ]
rc_list = [
    [0,0],[0,1],[0,2],[0,3],[0,4],
    [1,0],[1,1],[1,2],
    [2,0],[2,1],[2,2],[2,3],[2,4],
]

# ------------------------------------------------------------------------
# 2. Create a single figure with 4 subplots (1 row, 4 columns)
# ------------------------------------------------------------------------
fig, axs = plt.subplots(nrows=9, ncols=5, 
                        figsize=(8*5, 6*9), 
                        gridspec_kw={'hspace': 0.25},
                        sharex=True, sharey=True)  # wide figure

# ------------------------------------------------------------------------
# Subplot 1: TRAIN LOSS (IID)
# ------------------------------------------------------------------------
for K, (table_length, title_sub, (r,c)) in enumerate(zip(table_length_list,title_sub_list,rc_list)):
    
    split        = f'train'
    icl_sampling = 'iid'
    sample_ratio = '[0.200, 0.200, 0.200, 0.200, 0.200]'
    y_value      = f'acc_z_{table_length}'
    x_label      = x_label_name
    
    x_list = [x for x in np.arange(512//2*3+1) if x % 32 == 0][1:]
    for model in models.keys():
        ys_list = []
        for x in x_list:
            ys = []
            for random_seed in random_seeds:
                with open(f'../../saved4plot/FourGeneralization/{setup}/model={model} seed={random_seed}/{x}.pkl', 'rb') as f:
                    data = pickle.load(f)
                y = data[f"{split}-{icl_sampling}{sample_ratio}/{y_value}"]
                ys.append(to_python_float(y))
            ys_list.append(ys)
                
        mean_list = [np.mean(group) for group in ys_list]
        max_list  = [np.max (group) for group in ys_list]
        min_list  = [np.min (group) for group in ys_list]
        
        axs[3*r+0][c].plot(
            x_list,
            mean_list,
            #color='tab:blue',
            #marker='o',
            linestyle='-',
            lw=LW,
            label=models[model]
        )
        
        # Fill between (mean - var) and (mean + var) to represent "confidence interval"
        axs[3*r+0][c].fill_between(
            x_list,
            max_list,
            min_list,
            #color='tab:blue',
            alpha=0.2
        )
                
    #axs[3*r+1][c].set_title(f'$|\mathcal{{H}}|=$ {table_length} ({title_sub})', fontsize=title_fontsize)
    axs[3*r+0][c].set_title(f'$|\mathcal{{H}}|=$ {table_length} (Train)', fontsize=title_fontsize)
    
    axs[3*r+0][c].set_xlim(0, 768)
    axs[3*r+0][c].set_ylim(0.00, 1.00)
    
    axs[3*r+0][c].set_xticks([0, 256, 512, 768])
    axs[3*r+0][c].set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])
    
    axs[3*r+0][c].tick_params(axis='x', labelsize=tick_fontsize)
    axs[3*r+0][c].tick_params(axis='y', labelsize=tick_fontsize)
    
    
    split        = f'testI'
    icl_sampling = 'optimal'
    sample_ratio = '[None]'
    y_value      = f'acc_z_{table_length}'
    x_label      = x_label_name
    
    x_list = [x for x in np.arange(512//2*3+1) if x % 32 == 0]
    for model in models.keys():
        ys_list = []
        for x in x_list:
            ys = []
            for random_seed in random_seeds:
                with open(f'../../saved4plot/FourGeneralization/{setup}/model={model} seed={random_seed}/{x}.pkl', 'rb') as f:
                    data = pickle.load(f)
                y = data[f"{split}-{icl_sampling}{sample_ratio}/{y_value}"]
                ys.append(to_python_float(y))
            ys_list.append(ys)
                
        mean_list = [np.mean(group) for group in ys_list]
        max_list  = [np.max (group) for group in ys_list]
        min_list  = [np.min (group) for group in ys_list]
        
        axs[3*r+1][c].plot(
            x_list,
            mean_list,
            #color='tab:blue',
            #marker='o',
            linestyle='-',
            lw=LW,
            label=models[model]
        )
        
        # Fill between (mean - var) and (mean + var) to represent "confidence interval"
        axs[3*r+1][c].fill_between(
            x_list,
            max_list,
            min_list,
            #color='tab:blue',
            alpha=0.2
        )
    
    #axs[3*r+1][c].set_title(f'$|\mathcal{{H}}|=$ {table_length} ({title_sub})', fontsize=title_fontsize)
    axs[3*r+1][c].set_title(f'$|\mathcal{{H}}|=$ {table_length} ({title_sub})', fontsize=title_fontsize)
    
    axs[3*r+1][c].set_xlim(0, 768)
    axs[3*r+1][c].set_ylim(0.00, 1.00)
    
    axs[3*r+1][c].set_xticks([0, 256, 512, 768])
    axs[3*r+1][c].set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])
    
    axs[3*r+1][c].tick_params(axis='x', labelsize=tick_fontsize)
    axs[3*r+1][c].tick_params(axis='y', labelsize=tick_fontsize)



    
    
    split        = f'testO'
    icl_sampling = 'optimal'
    sample_ratio = '[None]'
    y_value      = f'acc_z_{table_length}'
    x_label      = x_label_name
    
    x_list = [x for x in np.arange(512//2*3+1) if x % 32 == 0]
    for model in models.keys():
        ys_list = []
        for x in x_list:
            ys = []
            for random_seed in random_seeds:
                with open(f'../../saved4plot/FourGeneralization/{setup}/model={model} seed={random_seed}/{x}.pkl', 'rb') as f:
                    data = pickle.load(f)
                y = data[f"{split}-{icl_sampling}{sample_ratio}/{y_value}"]
                ys.append(to_python_float(y))
            ys_list.append(ys)
                
        mean_list = [np.mean(group) for group in ys_list]
        max_list  = [np.max (group) for group in ys_list]
        min_list  = [np.min (group) for group in ys_list]
        
        axs[3*r+2][c].plot(
            x_list,
            mean_list,
            #color='tab:blue',
            #marker='o',
            linestyle='-',
            lw=LW,
            label=models[model]
        )
        
        # Fill between (mean - var) and (mean + var) to represent "confidence interval"
        axs[3*r+2][c].fill_between(
            x_list,
            max_list,
            min_list,
            #color='tab:blue',
            alpha=0.2
        )
    
    #axs[3*r+2][c].set_title(f'$|\mathcal{{H}}|=$ {table_length} ({title_sub})', fontsize=title_fontsize)
    axs[3*r+2][c].set_title(f'$|\mathcal{{H}}|=$ {table_length} ({title_sub})', fontsize=title_fontsize)
    
    axs[3*r+2][c].set_xlim(0, 768)
    axs[3*r+2][c].set_ylim(0.00, 1.00)
    
    axs[3*r+2][c].set_xticks([0, 256, 512, 768])
    axs[3*r+2][c].set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])
    
    axs[3*r+2][c].tick_params(axis='x', labelsize=tick_fontsize)
    axs[3*r+2][c].tick_params(axis='y', labelsize=tick_fontsize)


    if c == 0:
        axs[3*r+0][c].set_ylabel('Acc on $z$\n Train, iid', fontsize=y_label_fontsize)
        axs[3*r+1][c].set_ylabel('Acc on $z$\n Test(ID), Opt-T', fontsize=y_label_fontsize)
        axs[3*r+2][c].set_ylabel('Acc on $z$\n Test(OOD), Opt-T', fontsize=y_label_fontsize)

            

'''
axs[3][2].axis("off")
axs[3][1].legend(
    fontsize=legend_fontsize,
    #labelspacing=0.2,
    bbox_to_anchor=(2.05, 1.00)
)
'''
axs[4][2].legend(
    fontsize=legend_fontsize,
    #labelspacing=0.2,
    bbox_to_anchor=(2.50, 0.95)
)
axs[3][3].axis("off")
axs[4][3].axis("off")
axs[5][3].axis("off")
axs[3][4].axis("off")
axs[4][4].axis("off")
axs[5][4].axis("off")
#plt.tight_layout()  # helps prevent label overlap
print(f"FIG_{setup}/multiple_models_for_IOS_9x5.pdf")
plt.savefig(f"FIG_{setup}/multiple_models_for_IOS_9x5.pdf")
#plt.savefig(f"FIG_{setup}/multiple_curves_for_{key}S_4x3.png")
plt.show()
plt.close()