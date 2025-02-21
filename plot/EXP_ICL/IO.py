import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pickle
import numpy as np
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
size = 12

random_seeds   = [2023, 2024, 2025, 2026]

models = {
    'dual': 'Transformer',
}

modes = {
    'IOHypothesis_h+xy': 'ICL w/ instruction',
    'IOHypothesis_xy': 'ICL w/o instruction',
}

# Different model architecture
icl_k_list = [i for i in range(1,13)]

folder_name = 'FIG'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# ------------------------------------------------------------------------
# 2. Create a single figure with 4 subplots (1 row, 4 columns)
# ------------------------------------------------------------------------
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))  # wide figure

# ------------------------------------------------------------------------
# Subplot 1: TRAIN ACC (IID)
# ------------------------------------------------------------------------
# Directory and file naming setup
title_str      = 'Training Curves'
x_label_name   = 'Position in In-Context Sequence'
x_sticks       = [0,4,8,12]
x_labels       = [f"${i}$" for i in x_sticks]
var_multiplier = 1

axs_idx = 0
split        = 'train'
icl_sampling = 'iid'
sample_ratio = '[0.200, 0.200, 0.200, 0.200, 0.200]'
x_label      = x_label_name
y_label      = 'Accuracy on $y$ (i.i.d.)'


epoch = 768
for mode in modes.keys():
    ys_list = []
    for icl_k in icl_k_list:
        ys = []
        for random_seed in random_seeds:
            with open(f'../../saved4plot/ICL/{mode}/content={mode[13:]} seed={random_seed}/{epoch}.pkl', 'rb') as f:
                data = pickle.load(f)
            y = data[f"{split}-{icl_sampling}{sample_ratio}_icl/pos{icl_k}"].item()
            ys.append(y)
        ys_list.append(ys)

    mean_list = [np.mean(group) for group in ys_list]
    max_list  = [np.max (group) for group in ys_list]
    min_list  = [np.min (group) for group in ys_list]
    
    axs[axs_idx].plot(
        [x-1 for x in icl_k_list],
        mean_list,
        #color='tab:blue',
        #marker='o',
        linestyle='-',
        lw=LW,
        label=modes[mode]
    )
    
    # Fill between (mean - var) and (mean + var) to represent "confidence interval"
    axs[axs_idx].fill_between(
        [x-1 for x in icl_k_list],
        max_list,
        min_list,
        #color='tab:blue',
        alpha=0.2
    )
    
axs[axs_idx].set_xlabel(x_label, fontsize=x_label_fontsize)
axs[axs_idx].set_ylabel(y_label, fontsize=y_label_fontsize)
axs[axs_idx].set_title(title_str, fontsize=title_fontsize)

axs[axs_idx].set_xlim(0, 12)
axs[axs_idx].set_ylim(0.50, 1.00)

axs[axs_idx].set_xticks(x_sticks)
axs[axs_idx].set_xticklabels(x_labels)
axs[axs_idx].set_yticks([0.50, 0.60, 0.70, 0.80, 0.90, 1.00])

axs[axs_idx].tick_params(axis='x', labelsize=tick_fontsize)
axs[axs_idx].tick_params(axis='y', labelsize=tick_fontsize)

axs[axs_idx].legend(loc='lower right', fontsize=legend_fontsize)



# ------------------------------------------------------------------------
# Subplot 1: TRAIN ACC (IID)
# ------------------------------------------------------------------------
# Directory and file naming setup
title_str      = 'Testing Curves (ID Hypothesis)'
x_label_name   = 'Position in In-Context Sequence'
x_sticks       = [0,4,8,12]
x_labels       = [f"${i}$" for i in x_sticks]
var_multiplier = 1

axs_idx = 1
split        = 'testI'
icl_sampling = 'iid'
sample_ratio = '[0.200, 0.200, 0.200, 0.200, 0.200]'
x_label      = x_label_name
y_label      = 'Accuracy on $y$ (i.i.d.)'


epoch = 768
for mode in modes.keys():
    ys_list = []
    for icl_k in icl_k_list:
        ys = []
        for random_seed in random_seeds:
            with open(f'../../saved4plot/ICL/{mode}/content={mode[13:]} seed={random_seed}/{epoch}.pkl', 'rb') as f:
                data = pickle.load(f)
            y = data[f"{split}-{icl_sampling}{sample_ratio}_icl/pos{icl_k}"].item()
            ys.append(y)
        ys_list.append(ys)

    mean_list = [np.mean(group) for group in ys_list]
    max_list  = [np.max (group) for group in ys_list]
    min_list  = [np.min (group) for group in ys_list]
    
    axs[axs_idx].plot(
        [x-1 for x in icl_k_list],
        mean_list,
        #color='tab:blue',
        #marker='o',
        linestyle='-',
        lw=LW,
        label=modes[mode]
    )
    
    # Fill between (mean - var) and (mean + var) to represent "confidence interval"
    axs[axs_idx].fill_between(
        [x-1 for x in icl_k_list],
        max_list,
        min_list,
        #color='tab:blue',
        alpha=0.2
    )
    
axs[axs_idx].set_xlabel(x_label, fontsize=x_label_fontsize)
axs[axs_idx].set_ylabel(y_label, fontsize=y_label_fontsize)
axs[axs_idx].set_title(title_str, fontsize=title_fontsize)

axs[axs_idx].set_xlim(0, 12)
axs[axs_idx].set_ylim(0.50, 1.00)

axs[axs_idx].set_xticks(x_sticks)
axs[axs_idx].set_xticklabels(x_labels)
axs[axs_idx].set_yticks([0.50, 0.60, 0.70, 0.80, 0.90, 1.00])

axs[axs_idx].tick_params(axis='x', labelsize=tick_fontsize)
axs[axs_idx].tick_params(axis='y', labelsize=tick_fontsize)

#axs[axs_idx].legend(loc='lower right', fontsize=legend_fontsize)



# ------------------------------------------------------------------------
# 3. Finalize and Save the Single Figure
# ------------------------------------------------------------------------
plt.tight_layout()  # helps prevent label overlap
plt.savefig(f"FIG/ICL.pdf")
#plt.savefig(f"FIG/ICL.png")
plt.show()
plt.close()
