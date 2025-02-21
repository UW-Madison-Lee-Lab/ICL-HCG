import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pickle
import torch
import numpy as np
import os
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Font sizes
FS = 28
x_label_fontsize = FS + 8
y_label_fontsize = FS + 8
title_fontsize   = FS + 8
legend_fontsize  = FS + 4
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
      2,3,
    4,5,6,
    7,8,9,
    10,11,12
]
title_sub_list = [
                           r'$\boldsymbol{OOS}$', r'$\boldsymbol{OOS}$',
    r'$\boldsymbol{OOS}$', r'$\boldsymbol{OOS}$', r'$\boldsymbol{OOS}$',
    r'$\boldsymbol{IS}$' , r'$\boldsymbol{IS}$' , r'$\boldsymbol{IS}$' ,
    r'$\boldsymbol{OOS}$', r'$\boldsymbol{OOS}$', r'$\boldsymbol{OOS}$',
    ]
rc_list = [
          [3,1],[3,0],
    [2,2],[2,1],[2,0],
    [1,2],[1,1],[1,0],
    [0,2],[0,1],[0,0],
]

for key in ['I', 'O']:
    # ------------------------------------------------------------------------
    # 2. Create a single figure with 4 subplots (1 row, 4 columns)
    # ------------------------------------------------------------------------
    fig, axs = plt.subplots(nrows=4, ncols=3, 
                            figsize=(20, 18), 
                            gridspec_kw={'hspace': 0.25},
                            sharex=True, sharey=True)  # wide figure
    
    # ------------------------------------------------------------------------
    # Subplot 1: TRAIN LOSS (IIS)
    # ------------------------------------------------------------------------
    split        = f'test{key}'
    icl_sampling = 'optimal'
    
    y_value      = 'acc_z'
    x_label      = x_label_name
    y_label      = 'Acc on $z$ (Opt-T)'
    title_str    = 'Training Loss'
    
    for K, (table_length, title_sub, (r,c)) in enumerate(zip(table_length_list,title_sub_list,rc_list)):
        if 1:
            for idx, random_seed in enumerate(random_seed_list):
                x_list = [x for x in np.arange(512//2*3+1) if x % 32 == 0][1:]
                y_list = []
                seed = random_seed
                for x in x_list:
                    with open(f'../../saved4plot/FourGeneralization/{setup}/model={model} seed={seed}/{x}.pkl', 'rb') as f:
                        data = pickle.load(f)
                        y = data[f"{split}-{icl_sampling}[None]/{y_value}_{table_length}"].item()
                        y_list.append(y)
                axs[r][c].plot(
                    x_list,
                    y_list,
                    label=labels[idx],
                    linewidth=LW,
                    zorder=10
                )
        
        if c == 0:
            axs[r][c].set_ylabel(y_label, fontsize=y_label_fontsize)
        #axs[r][c].set_title(f'$|\mathcal{{H}}|=$ {table_length} ({title_sub})', fontsize=title_fontsize)
        axs[r][c].set_title(f'$|\mathcal{{H}}|=$ {table_length} ({title_sub})', fontsize=title_fontsize)
        
        axs[r][c].set_xlim(0, 768)
        axs[r][c].set_ylim(0.00, 1.00)
        
        axs[r][c].set_xticks([0, 256, 512, 768])
        axs[r][c].set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])
        
        axs[r][c].tick_params(axis='x', labelsize=tick_fontsize)
        axs[r][c].tick_params(axis='y', labelsize=tick_fontsize)
    '''
    axs[0].legend(
        fontsize=legend_fontsize,
        ncol=4,
        loc='upper center',
        columnspacing=1.0,
        bbox_to_anchor=(2.9, 1.35)
    )
    '''
    axs[3][2].axis("off")
    axs[3][1].legend(
        fontsize=legend_fontsize,
        #labelspacing=0.2,
        bbox_to_anchor=(2.05, 1.00)
    )
    
    #plt.tight_layout()  # helps prevent label overlap
    print(f"FIG_{setup}/multiple_curves_for_{key}_S_4x3.pdf")
    plt.savefig(f"FIG_{setup}/multiple_curves_for_{key}S_4x3.pdf")
    #plt.savefig(f"FIG_{setup}/multiple_curves_for_{key}S_4x3.png")
    plt.show()
    plt.close()