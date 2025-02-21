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

# Number of runs
num_runs = 4

# Labels for each run
labels = [
    r"$1^{\text{st}}$" + " run",
    r"$2^{\text{nd}}$" + " run",
    r"$3^{\text{rd}}$" + " run",
    r"$4^{\text{th}}$" + " run",
]

model = 'transformer'

setup = 'IOHypothesis+Size_h+xy+z'
step_name      = 'global_step'
x_label_name   = 'Epoch'

random_seed_list = [2023,2024,2025,2026]
table_length_list = [
    14,13,12,11,10,
     9, 8, 7,
     6, 5, 4, 3, 2,
]
title_sub_list = [
    r'$\boldsymbol{OOS}$', r'$\boldsymbol{OOS}$', r'$\boldsymbol{OOS}$', r'$\boldsymbol{OOS}$', r'$\boldsymbol{OOS}$',
    r'$\boldsymbol{IS}$' , r'$\boldsymbol{IS}$' , r'$\boldsymbol{IS}$' ,
    r'$\boldsymbol{OOS}$', r'$\boldsymbol{OOS}$', r'$\boldsymbol{OOS}$', r'$\boldsymbol{OOS}$', r'$\boldsymbol{OOS}$',
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
# Subplot 1: TRAIN LOSS (IIS)
# ------------------------------------------------------------------------
for K, (table_length, title_sub, (r,c)) in enumerate(zip(table_length_list,title_sub_list,rc_list)):
    
    split        = f'train'
    icl_sampling = 'iid'
    sample_ratio = '[0.200, 0.200, 0.200, 0.200, 0.200]'
    y_value      = 'acc_z'
    x_label      = x_label_name
    
    for idx, random_seed in enumerate(random_seed_list):
        x_list = [x for x in np.arange(512//2*3+1) if x % 32 == 0][1:]
        y_list = []
        seed = random_seed
        for x in x_list:
            with open(f'../../saved4plot/FourGeneralization/{setup}/model={model} seed={seed}/{x}.pkl', 'rb') as f:
                data = pickle.load(f)
                y = data[f"{split}-{icl_sampling}{sample_ratio}/{y_value}_{table_length}"]
                y_list.append(to_python_float(y))
        axs[3*r+0][c].plot(
            x_list,
            y_list,
            label=labels[idx],
            linewidth=LW,
            zorder=10
        )
    #axs[3*r+1][c].set_title(f'$|\mathcal{{H}}|=$ {table_length} ({title_sub})', fontsize=title_fontsize)
    axs[3*r+0][c].set_title(f'$|\mathcal{{H}}|=$ {table_length} ({title_sub})', fontsize=title_fontsize)
    
    axs[3*r+0][c].set_xlim(0, 768)
    axs[3*r+0][c].set_ylim(0.00, 1.00)
    
    axs[3*r+0][c].set_xticks([0, 256, 512, 768])
    axs[3*r+0][c].set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])
    
    axs[3*r+0][c].tick_params(axis='x', labelsize=tick_fontsize)
    axs[3*r+0][c].tick_params(axis='y', labelsize=tick_fontsize)
    
    
    split        = f'testI'
    icl_sampling = 'optimal'
    sample_ratio = '[None]'
    y_value      = 'acc_z'
    x_label      = x_label_name
    
    for idx, random_seed in enumerate(random_seed_list):
        x_list = [x for x in np.arange(512//2*3+1) if x % 32 == 0]
        y_list = []
        seed = random_seed
        for x in x_list:
            with open(f'../../saved4plot/FourGeneralization/{setup}/model={model} seed={seed}/{x}.pkl', 'rb') as f:
                data = pickle.load(f)
                y = data[f"{split}-{icl_sampling}{sample_ratio}/{y_value}_{table_length}"].cpu().item()
                y_list.append(to_python_float(y))
        axs[3*r+1][c].plot(
            x_list,
            y_list,
            label=labels[idx],
            linewidth=LW,
            zorder=10
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
    y_value      = 'acc_z'
    x_label      = x_label_name
    
    for idx, random_seed in enumerate(random_seed_list):
        x_list = [x for x in np.arange(512//2*3+1) if x % 32 == 0]
        y_list = []
        seed = random_seed
        for x in x_list:
            with open(f'../../saved4plot/FourGeneralization/{setup}/model={model} seed={seed}/{x}.pkl', 'rb') as f:
                data = pickle.load(f)
                y = data[f"{split}-{icl_sampling}{sample_ratio}/{y_value}_{table_length}"].cpu().item()
                y_list.append(to_python_float(y))
        axs[3*r+2][c].plot(
            x_list,
            y_list,
            label=labels[idx],
            linewidth=LW,
            zorder=10
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
        axs[3*r+0][c].set_ylabel('Acc on $z$ (i.i.d.)\n Train', fontsize=y_label_fontsize)
        axs[3*r+1][c].set_ylabel('Acc on $z$ (Opt-T)\n Test', fontsize=y_label_fontsize)
        axs[3*r+2][c].set_ylabel('Acc on $z$ (Opt-T)\n Test', fontsize=y_label_fontsize)

            

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
print(f"FIG_{setup}/multiple_curves_for_IOS_9x5.pdf")
plt.savefig(f"FIG_{setup}/multiple_curves_for_IOS_9x5.pdf")
#plt.savefig(f"FIG_{setup}/multiple_curves_for_{key}S_4x3.png")
plt.show()
plt.close()