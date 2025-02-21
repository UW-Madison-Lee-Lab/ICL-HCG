import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle
import os
# ------------------------------------------------------------------------
# 1. General Matplotlib & LaTeX Setup
# ------------------------------------------------------------------------
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Font sizes
FS = 28
x_label_fontsize = FS + 8
y_label_fontsize = FS + 8
title_fontsize   = FS + 8
legend_fontsize  = FS
tick_fontsize    = FS

# Custom line width
LW = 6  # thick lines
S=15

# Different model architecture
models = {
    'transformer': 'Transformer',
    'mamba': 'Mamba',
    'lstm': 'GRU', 
    'gru': 'LSTM'
    }

# Directory and file naming setup
setup          = 'IOHypothesis_h+xy+z'
step_name      = 'global_step'
x_label_name   = '\#Training Classes ($N^{\\text{train}}$)'
table_length   = 8
DP             = '[0.250, 0.250, 0.250, 0.250]'
x_list         = [1,4,16,64,256,1024,4096,12358]
x_sticks       = [1,4,16,64,256,1024,4096,12358]
x_labels       = [f"$2^{{{i}}}$" for i in [0,2,4,6,8,10,12,14]]
random_seeds   = [2023, 2024, 2025, 2026]
var_multiplier = 1
epoch=768


folder_name = 'FIG'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    
    
for key in ['I','O']:
    plt.figure(figsize=(8, 6))
    
    axs_idx = 2
    split        = 'test'+key
    icl_sampling = 'optimal[None]'
    y_value      = 'acc_z'
    x_label      = x_label_name
    y_label      = 'Acc on $z$ (Opt-T)'
    if key == 'I':
        title_str = 'Testing Curves on ID Hypotheses'
    if key == 'O':
        title_str = 'Testing Curves on OOD Hypotheses'
    
    for model in models.keys():
        x2y_list = {x:None for x in x_list}
        for x in x_list:
            y_list = []
            for random_seed in random_seeds:
                with open(f'../../saved4plot/NUMTRAIN/{setup}/model={model} num={x} seed={random_seed}/{epoch}.pkl', 'rb') as f:
                    data = pickle.load(f)
                y = data[f"{split}-{icl_sampling}/{y_value}_{table_length}"].item()
                y_list.append(y)
            y_list.sort()
            x2y_list[x] = y_list#[1:3]
                    
        mean_list = [np.mean(y_list) for x, y_list in x2y_list.items()]
        max_list  = [np.max (y_list) for x, y_list in x2y_list.items()]
        min_list  = [np.min (y_list) for x, y_list in x2y_list.items()]
    
        plt.plot(
            x_list,
            mean_list,
            #color='tab:blue',
            marker='o',
            markersize=S,
            linestyle='-',
            linewidth=LW,
            label=models[model]
        )
        
        # Fill between (mean - var) and (mean + var) to represent "confidence interval"
        plt.fill_between(
            x_list,
            max_list,
            min_list,
            #color='tab:blue',
            alpha=0.2
        )
        
    plt.xlabel(x_label, fontsize=x_label_fontsize)
    plt.ylabel(y_label, fontsize=y_label_fontsize)
    plt.title(title_str, fontsize=title_fontsize)
    
    plt.xlim(1, 1024)
    plt.ylim(0.00, 1.00)
    
    plt.xscale('log')
    plt.xticks(x_sticks, x_labels)
    plt.yticks([0.00, 0.25, 0.50, 0.75, 1.00])
    
    plt.tick_params(axis='x', labelsize=tick_fontsize)
    plt.tick_params(axis='y', labelsize=tick_fontsize)
    
    plt.legend(fontsize=legend_fontsize)


    plt.tight_layout()  # helps prevent label overlap
    plt.savefig(f"FIG/num_train_{key}_1x1.pdf")
    #plt.savefig(f"FIG/{setup}_multiple_curves_1x5_combined.png")
    plt.show()
    plt.close()


