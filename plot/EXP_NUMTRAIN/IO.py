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
HEAD           = 'TableGeneralization'
Generalization = 'IOHypothesis_h+xy+z'
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
    
    
# ------------------------------------------------------------------------
# 2. Create a single figure with 4 subplots (1 row, 4 columns)
# ------------------------------------------------------------------------
fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(40, 6))  # wide figure

# ------------------------------------------------------------------------
# Subplot 1: TRAIN LOSS (IID)
# ------------------------------------------------------------------------
axs_idx = 0
split        = 'train'
icl_sampling = 'iid[0.200, 0.200, 0.200, 0.200, 0.200]'
y_value      = 'acc_z'
x_label      = x_label_name
y_label      = 'Acc on $z$ (i.i.d.)'
title_str    = 'Training Curves on ID Hypotheses'

for model in models.keys():
    x2y_list = {x:None for x in x_list}
    for x in x_list:
        y_list = []
        for random_seed in random_seeds:
            with open(f'../../saved4plot/NUMTRAIN/{Generalization}/model={model} num={x} seed={random_seed}/{epoch}.pkl', 'rb') as f:
                data = pickle.load(f)
            y = data[f"{split}-{icl_sampling}/{y_value}_{table_length}"].item()
            y_list.append(y)
        y_list.sort()
        x2y_list[x] = y_list#[1:3]
                
    mean_list = [np.mean(y_list) for x, y_list in x2y_list.items()]
    max_list  = [np.max (y_list) for x, y_list in x2y_list.items()]
    min_list  = [np.min (y_list) for x, y_list in x2y_list.items()]

    axs[axs_idx].plot(
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
    axs[axs_idx].fill_between(
        x_list,
        max_list,
        min_list,
        #color='tab:blue',
        alpha=0.2
    )
    
axs[axs_idx].set_xlabel(x_label, fontsize=x_label_fontsize)
axs[axs_idx].set_ylabel(y_label, fontsize=y_label_fontsize)
axs[axs_idx].set_title(title_str, fontsize=title_fontsize)

axs[axs_idx].set_xlim(1, 2**14)
axs[axs_idx].set_ylim(0.00, 1.00)

axs[axs_idx].set_xscale('log')
axs[axs_idx].set_xticks(x_sticks)
axs[axs_idx].set_xticklabels(x_labels)
axs[axs_idx].set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])

axs[axs_idx].tick_params(axis='x', labelsize=tick_fontsize)
axs[axs_idx].tick_params(axis='y', labelsize=tick_fontsize)

axs[axs_idx].legend(loc='lower left', fontsize=legend_fontsize)


# ------------------------------------------------------------------------
# Subplot 2: TEST ACCURACY (IID)
# ------------------------------------------------------------------------
axs_idx = 1
split        = 'testI'
icl_sampling = 'iid[0.200, 0.200, 0.200, 0.200, 0.200]'
y_value      = 'acc_z'
x_label      = x_label_name
y_label      = 'Acc on $z$ (i.i.d.)'
title_str    = 'Testing Curves on ID Hypotheses'

for model in models.keys():
    x2y_list = {x:None for x in x_list}
    for x in x_list:
        y_list = []
        for random_seed in random_seeds:
            with open(f'../../saved4plot/NUMTRAIN/{Generalization}/model={model} num={x} seed={random_seed}/{epoch}.pkl', 'rb') as f:
                data = pickle.load(f)
            y = data[f"{split}-{icl_sampling}/{y_value}_{table_length}"].item()
            y_list.append(y)
        y_list.sort()
        x2y_list[x] = y_list#[1:3]
                
    mean_list = [np.mean(y_list) for x, y_list in x2y_list.items()]
    max_list  = [np.max (y_list) for x, y_list in x2y_list.items()]
    min_list  = [np.min (y_list) for x, y_list in x2y_list.items()]

    axs[axs_idx].plot(
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
    axs[axs_idx].fill_between(
        x_list,
        max_list,
        min_list,
        #color='tab:blue',
        alpha=0.2
    )
    
axs[axs_idx].set_xlabel(x_label, fontsize=x_label_fontsize)
axs[axs_idx].set_ylabel(y_label, fontsize=y_label_fontsize)
axs[axs_idx].set_title(title_str, fontsize=title_fontsize)

axs[axs_idx].set_xlim(1, 1024)
axs[axs_idx].set_ylim(0.00, 1.00)

axs[axs_idx].set_xscale('log')
axs[axs_idx].set_xticks(x_sticks)
axs[axs_idx].set_xticklabels(x_labels)
axs[axs_idx].set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])

axs[axs_idx].tick_params(axis='x', labelsize=tick_fontsize)
axs[axs_idx].tick_params(axis='y', labelsize=tick_fontsize)

#axs[axs_idx].legend(fontsize=legend_fontsize)


# ------------------------------------------------------------------------
# Subplot 3: TEST CONFIDENCE (IID)
# ------------------------------------------------------------------------
axs_idx = 2
split        = 'testI'
icl_sampling = 'optimal[None]'
y_value      = 'acc_z'
x_label      = x_label_name
y_label      = 'Acc on $z$ (Opt-T)'
title_str    = 'Testing Curves on ID Hypotheses'

for model in models.keys():
    x2y_list = {x:None for x in x_list}
    for x in x_list:
        y_list = []
        for random_seed in random_seeds:
            with open(f'../../saved4plot/NUMTRAIN/{Generalization}/model={model} num={x} seed={random_seed}/{epoch}.pkl', 'rb') as f:
                data = pickle.load(f)
            y = data[f"{split}-{icl_sampling}/{y_value}_{table_length}"].item()
            y_list.append(y)
        y_list.sort()
        x2y_list[x] = y_list#[1:3]
                
    mean_list = [np.mean(y_list) for x, y_list in x2y_list.items()]
    max_list  = [np.max (y_list) for x, y_list in x2y_list.items()]
    min_list  = [np.min (y_list) for x, y_list in x2y_list.items()]

    axs[axs_idx].plot(
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
    axs[axs_idx].fill_between(
        x_list,
        max_list,
        min_list,
        #color='tab:blue',
        alpha=0.2
    )
    
axs[axs_idx].set_xlabel(x_label, fontsize=x_label_fontsize)
axs[axs_idx].set_ylabel(y_label, fontsize=y_label_fontsize)
axs[axs_idx].set_title(title_str, fontsize=title_fontsize)

axs[axs_idx].set_xlim(1, 1024)
axs[axs_idx].set_ylim(0.00, 1.00)

axs[axs_idx].set_xscale('log')
axs[axs_idx].set_xticks(x_sticks)
axs[axs_idx].set_xticklabels(x_labels)
axs[axs_idx].set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])

axs[axs_idx].tick_params(axis='x', labelsize=tick_fontsize)
axs[axs_idx].tick_params(axis='y', labelsize=tick_fontsize)

#axs[axs_idx].legend(fontsize=legend_fontsize)


# ------------------------------------------------------------------------
# Subplot 4: TEST ACCURACY (IID)
# ------------------------------------------------------------------------
axs_idx = 3
split        = 'testO'
icl_sampling = 'iid[0.200, 0.200, 0.200, 0.200, 0.200]'
y_value      = 'acc_z'
x_label      = x_label_name
y_label      = 'Acc on $z$ (i.i.d.)'
title_str    = 'Testing Curves on OOD Hypotheses'

for model in models.keys():
    x2y_list = {x:None for x in x_list}
    for x in x_list:
        y_list = []
        for random_seed in random_seeds:
            with open(f'../../saved4plot/NUMTRAIN/{Generalization}/model={model} num={x} seed={random_seed}/{epoch}.pkl', 'rb') as f:
                data = pickle.load(f)
            y = data[f"{split}-{icl_sampling}/{y_value}_{table_length}"].item()
            y_list.append(y)
        y_list.sort()
        x2y_list[x] = y_list#[1:3]
                
    mean_list = [np.mean(y_list) for x, y_list in x2y_list.items()]
    max_list  = [np.max (y_list) for x, y_list in x2y_list.items()]
    min_list  = [np.min (y_list) for x, y_list in x2y_list.items()]

    axs[axs_idx].plot(
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
    axs[axs_idx].fill_between(
        x_list,
        max_list,
        min_list,
        #color='tab:blue',
        alpha=0.2
    )
    
axs[axs_idx].set_xlabel(x_label, fontsize=x_label_fontsize)
axs[axs_idx].set_ylabel(y_label, fontsize=y_label_fontsize)
axs[axs_idx].set_title(title_str, fontsize=title_fontsize)

axs[axs_idx].set_xlim(1, 1024)
axs[axs_idx].set_ylim(0.00, 1.00)

axs[axs_idx].set_xscale('log')
axs[axs_idx].set_xticks(x_sticks)
axs[axs_idx].set_xticklabels(x_labels)
axs[axs_idx].set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])

axs[axs_idx].tick_params(axis='x', labelsize=tick_fontsize)
axs[axs_idx].tick_params(axis='y', labelsize=tick_fontsize)

#axs[axs_idx].legend(fontsize=legend_fontsize)


# ------------------------------------------------------------------------
# Subplot 5: TEST CONFIDENCE (IID)
# ------------------------------------------------------------------------
axs_idx = 4
split        = 'testO'
icl_sampling = 'optimal[None]'
y_value      = 'acc_z'
x_label      = x_label_name
y_label      = 'Acc on $z$ (Opt-T)'
title_str    = 'Testing Curves on OOD Hypotheses'

for model in models.keys():
    x2y_list = {x:None for x in x_list}
    for x in x_list:
        y_list = []
        for random_seed in random_seeds:
            with open(f'../../saved4plot/NUMTRAIN/{Generalization}/model={model} num={x} seed={random_seed}/{epoch}.pkl', 'rb') as f:
                data = pickle.load(f)
            y = data[f"{split}-{icl_sampling}/{y_value}_{table_length}"].item()
            y_list.append(y)
        y_list.sort()
        x2y_list[x] = y_list#[1:3]
                
    mean_list = [np.mean(y_list) for x, y_list in x2y_list.items()]
    max_list  = [np.max (y_list) for x, y_list in x2y_list.items()]
    min_list  = [np.min (y_list) for x, y_list in x2y_list.items()]

    axs[axs_idx].plot(
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
    axs[axs_idx].fill_between(
        x_list,
        max_list,
        min_list,
        #color='tab:blue',
        alpha=0.2
    )
    
axs[axs_idx].set_xlabel(x_label, fontsize=x_label_fontsize)
axs[axs_idx].set_ylabel(y_label, fontsize=y_label_fontsize)
axs[axs_idx].set_title(title_str, fontsize=title_fontsize)

axs[axs_idx].set_xlim(1, 1024)
axs[axs_idx].set_ylim(0.00, 1.00)

axs[axs_idx].set_xscale('log')
axs[axs_idx].set_xticks(x_sticks)
axs[axs_idx].set_xticklabels(x_labels)
axs[axs_idx].set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])

axs[axs_idx].tick_params(axis='x', labelsize=tick_fontsize)
axs[axs_idx].tick_params(axis='y', labelsize=tick_fontsize)

#axs[axs_idx].legend(fontsize=legend_fontsize)



# ------------------------------------------------------------------------
# 3. Finalize and Save the Single Figure
# ------------------------------------------------------------------------
plt.tight_layout()  # helps prevent label overlap
plt.savefig(f"FIG/num_train_IO_1x5.pdf")
plt.show()
plt.close()


### back up
if 1:
    # Calculate means and variances for data1
    '''
    mean_list = [np.mean(group) for group in ys_list]
    std_list  = [np.std(group, ddof=1) for group in ys_list]

    axs[axs_idx].plot(
        x_sticks,
        mean_list,
        #color='tab:blue',
        marker='o',
        linestyle='-',
        label=models[model]
    )
    
    # Fill between (mean - var) and (mean + var) to represent "confidence interval"
    axs[axs_idx].fill_between(
        x_sticks,
        [m - v for m, v in zip(mean_list, std_list)],
        [m + v for m, v in zip(mean_list, std_list)],
        #color='tab:blue',
        alpha=0.2
    )
    '''