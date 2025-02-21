import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pickle
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
      2,3,
    4,5,6,
    7,8,9,
    10,11,12
]
title_sub_list = [
                           r'$\boldsymbol{OOS}$', r'$\boldsymbol{OOS}$',
    r'$\boldsymbol{OOS}$', r'$\boldsymbol{OOS}$', r'$\boldsymbol{OOS}$' ,
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
    sample_ratio = '[None]'
    
    for K, (table_length, title_sub, (r,c)) in enumerate(zip(table_length_list,title_sub_list,rc_list)):
        
        y_value      = f'acc_z_{table_length}'
        x_label      = x_label_name
        y_label      = 'Acc on $z$ (Opt-T)'
        
        if 1:
            x_list = [x for x in np.arange(512//2*3+1) if x % 32 == 0][1:]
            for model in models.keys():
                ys_list = []
                for x in x_list:
                    ys = []
                    for random_seed in random_seeds:
                        with open(f'../../saved4plot/FourGeneralization/{setup}/model={model} seed={random_seed}/{x}.pkl', 'rb') as f:
                            data = pickle.load(f)
                        y = data[f"{split}-{icl_sampling}{sample_ratio}/{y_value}"].item()
                        ys.append(y)
                    ys_list.append(ys)
                        
                mean_list = [np.mean(group) for group in ys_list]
                max_list  = [np.max (group) for group in ys_list]
                min_list  = [np.min (group) for group in ys_list]
                
                axs[r][c].plot(
                    x_list,
                    mean_list,
                    #color='tab:blue',
                    #marker='o',
                    linestyle='-',
                    lw=LW,
                    label=models[model]
                )
                
                # Fill between (mean - var) and (mean + var) to represent "confidence interval"
                axs[r][c].fill_between(
                    x_list,
                    max_list,
                    min_list,
                    #color='tab:blue',
                    alpha=0.2
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
    print(f"FIG_{setup}/multiple_models_for_{key}_S_4x3.pdf")
    plt.savefig(f"FIG_{setup}/multiple_models_for_{key}S_4x3.pdf")
    #plt.savefig(f"FIG_{setup}/multiple_curves_for_{key}S_4x3.png")
    plt.show()
    plt.close()