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

setup = 'IOHypothesis_h+xy+z'
step_name      = 'Step'
x_label_name   = 'Epoch'

sample_ratio = '[0.200, 0.200, 0.200, 0.200, 0.200]'

folder_name = f'FIG_{setup}'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

for key in ['I', 'O']:
    plt.figure(figsize=(8, 6))
    
    split           = 'test'+key
    icl_sampling    = 'optimal'
    y_value         = 'acc_z_8'
    x_label         = x_label_name
    y_label         = 'Acc on $z$ (Opt-T)'
    if key == 'I':
        title       = 'Testing Curve on ID Hypotheses'
    if key == 'O':
        title       = 'Testing Curve on OOD Hypotheses'

    x_list = [x for x in np.arange(512//2*3+1) if x % 32 == 0]
    for model in models.keys():
        ys_list = []
        for x in x_list:
            ys = []
            for random_seed in random_seeds:
                with open(f'../../saved4plot/FourGeneralization/{setup}/model={model} seed={random_seed}/{x}.pkl', 'rb') as f:
                    data = pickle.load(f)
                y = data[f"{split}-{icl_sampling}[None]/{y_value}"].item()
                ys.append(y)
            ys_list.append(ys)
        
        mean_list = [np.mean(group) for group in ys_list]
        max_list  = [np.max (group) for group in ys_list]
        min_list  = [np.min (group) for group in ys_list]
        
        plt.plot(
            x_list,
            mean_list,
            #color='tab:blue',
            #marker='o',
            linestyle='-',
            lw=LW,
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
    plt.title(title, fontsize=title_fontsize)
    
    plt.xlim(0, 768)
    plt.ylim(0.00, 1.00)
    
    plt.xticks([0, 256, 512, 768], fontsize=tick_fontsize)
    plt.yticks([0.00, 0.25, 0.50, 0.75, 1.00], fontsize=tick_fontsize)
    
    leg = plt.legend(fontsize=legend_fontsize,
               labelspacing=0.2,
               #bbox_to_anchor=(1.028, -0.043, 0.0, 0.0)
               loc = 'lower right')
    leg.set_zorder(10)
    plt.tight_layout()  # helps prevent label overlap
    plt.savefig(f"FIG_{setup}/multiple_models_for_{key}_1x1.pdf")
    #plt.savefig(f"FIG_{setup}/multiple_models_for_{key}_1x1.png")
    plt.show()
    plt.close()