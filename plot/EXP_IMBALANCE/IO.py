import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pickle
import numpy as np
import os
import sys
sys.path.insert(0, '..')
from utils import to_python_float

def convert(list_of_lists):
    # Find the longest inner list length
    max_len = max(len(sublist) for sublist in list_of_lists)
    
    result = []
    # For each index up to the longest length
    for i in range(max_len):
        row = []
        # Collect the i-th element from each inner list if it exists
        for sublist in list_of_lists:
            if i < len(sublist):
                row.append(sublist[i])
        result.append(row)
    
    return result

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
legend_fontsize  = FS + 4
tick_fontsize    = FS

# Custom line width
LW = 6  # thick lines

# Different model architecture
models = {
    'transformer': 'Transformer',
    }

random_seeds   = [2023, 2024, 2025, 2026]

setup          = 'IOHypothesis_h+xy+z'
x_label_name   = 'Epoch'
DP_list        = [1.0, 4.0, 9.0]

colors = [
    #"#b3cde0",  # lightest
    "#6497b1",
    "#005b96",
    "#03396c"   # darkest
][::-1]

linestyles = [
    #":",   # dotted
    "-.",  # dash-dot
    "--",  # dashed
    "-"    # solid
][::-1]

folder_name = f'FIG_{setup}'

if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    
# ------------------------------------------------------------------------
# 2. Create a single figure with 4 subplots (1 row, 4 columns)
# ------------------------------------------------------------------------
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6), sharey=True)  # wide figure
    
for c, key in enumerate(['I','O']):
    if key == 'I':
        title_str      = r'ID'+' Hypothesis Class Generalization'
    if key == 'O':
        title_str      = r'OOD'+' Hypothesis Class Generalization'
        
    split        = 'test'+key
    icl_sampling = 'optimal'
    y_value      = 'acc_z_8'
    x_label      = x_label_name
    y_label      = 'Acc on $z$ (Opt-T)'
    
    x_list = [x for x in np.arange(512//2*3+1) if x % 32 == 0]
    for model in models.keys():
        for DP, color, linestyle in zip(DP_list, colors, linestyles):
            ys_list = []
            for x in x_list:
                ys = []
                for random_seed in random_seeds:
                    with open(f'../../saved4plot/DP/{setup}/DP={DP} seed={random_seed}/{x}.pkl', 'rb') as f:
                        data = pickle.load(f)
                    y = data[f"{split}-{icl_sampling}[None]/{y_value}"].item()
                    ys.append(y)
                ys_list.append(ys)
            
            mean_list = [np.mean(group) for group in ys_list]
            max_list  = [np.max (group) for group in ys_list]
            min_list  = [np.min (group) for group in ys_list]
        
            axs[c].plot(
                x_list,
                mean_list,
                color=color,
                #marker='o',
                linestyle=linestyle,
                lw=LW,
                label=f'D={DP}'
            )
            
            # Fill between (mean - var) and (mean + var) to represent "confidence interval"
            axs[c].fill_between(
                x_list,
                max_list,
                min_list,
                color=color,
                alpha=0.2
            )
    
    axs[c].set_xlabel(x_label, fontsize=x_label_fontsize)
    if c == 0:
        axs[c].set_ylabel(y_label, fontsize=y_label_fontsize)
    axs[c].set_title(title_str, fontsize=title_fontsize)
    
    axs[c].set_xlim(0, 768)
    axs[c].set_ylim(0.0, 1.0)
    
    axs[c].set_xticks([0, 256, 512, 768])
    axs[c].set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])
    
    axs[c].tick_params(axis='x', labelsize=tick_fontsize)
    axs[c].tick_params(axis='y', labelsize=tick_fontsize)
    
    axs[c].legend(fontsize=legend_fontsize, loc='lower right')


'''
# ------------------------------------------------------------------------
# Subplot 2: OOD Generalization (Opt-T)
# ------------------------------------------------------------------------
# Directory and file naming setup
setup          = 'IOHypothesis_h+xy+z'
title_str      = r'OOD'+' Hypothesis Class Generalization'
step_name      = 'global_step'
x_label_name   = 'Epoch'
table_length   = 4
DP_list        = [1,4]

c = 1
split        = 'testO'
icl_sampling = 'optimal'
y_value      = 'acc_z_8'
x_label      = x_label_name
y_label      = 'Acc on $z$ (Opt-T)'

x_list = [x for x in np.arange(512//2*3+1) if x % 32 == 0]
for model in models.keys():
    for DP in DP_list:
        ys_list = []
        for random_seed in random_seeds:
            #data = df[f"{model} {DP} {random_seed} - {split}-{icl_sampling}[None]/{y_value}_{table_length}"]
            with open(f'../../saved4plot/DP/{Generalization}/model={model} num={x} seed={random_seed}/{epoch}.pkl', 'rb') as f:
                data = pickle.load(f)
            ys_list.append(list(data))
        ys_list = convert(ys_list)
        
        mean_list = [np.mean(group) for group in ys_list]
        max_list  = [np.max (group) for group in ys_list]
        min_list  = [np.min (group) for group in ys_list]
    
        axs[c].plot(
            list(df[step_name]),
            mean_list,
            #color='tab:blue',
            #marker='o',
            linestyle='-',
            lw=LW,
            label=f'Disparity={DP}'
        )
        
        # Fill between (mean - var) and (mean + var) to represent "confidence interval"
        axs[c].fill_between(
            list(df[step_name]),
            max_list,
            min_list,
            #color='tab:blue',
            alpha=0.2
        )

axs[c].set_xlabel(x_label, fontsize=x_label_fontsize)
#axs[c].set_ylabel(y_label, fontsize=y_label_fontsize)
axs[c].set_title(title_str, fontsize=title_fontsize)

axs[c].set_xlim(1, 512)
axs[c].set_ylim(0.0, 1.0)

axs[c].set_xticks([1, 128, 256, 374, 512])
axs[c].set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])

axs[c].tick_params(axis='x', labelsize=tick_fontsize)
axs[c].tick_params(axis='y', labelsize=tick_fontsize)
'''

#axs[0].legend(fontsize=legend_fontsize, loc='lower right')


# ------------------------------------------------------------------------
# 3. Finalize and Save the Single Figure
# ------------------------------------------------------------------------
plt.tight_layout()  # helps prevent label overlap
#plt.savefig(f"FIG_{setup}/multiple_models_for_IOS_9x5.png")
plt.savefig(f"FIG_{setup}/imbalance_for_IO_1x2.pdf")
plt.show()
plt.close()