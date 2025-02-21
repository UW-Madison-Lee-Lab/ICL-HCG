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

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(8*3, 6*2))  # wide figure

# ------------------------------------------------------------------------
# Subplot 0-0: TRAIN LOSS (IID)
# ------------------------------------------------------------------------
split        = 'train'
icl_sampling = 'iid'
sample_ratio = '[0.200, 0.200, 0.200, 0.200, 0.200]'
y_value      = 'loss__8'
x_label      = x_label_name
y_label      = 'Cross Entropy Loss on $s$'
title_str    = 'Training Curves on ID Hypotheses'
r, c = 0, 0

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
    
#axs[r][c].set_xlabel(x_label, fontsize=x_label_fontsize)
axs[r][c].set_ylabel(y_label, fontsize=y_label_fontsize)
axs[r][c].set_title(title_str, fontsize=title_fontsize)

axs[r][c].set_xlim(0, 768)
axs[r][c].set_ylim(0.25, 0.31)

axs[r][c].set_xticks([0, 256, 512, 768])
axs[r][c].set_yticks([0.25, 0.27, 0.29, 0.31])

axs[r][c].tick_params(axis='x', labelsize=tick_fontsize)
axs[r][c].tick_params(axis='y', labelsize=tick_fontsize)

axs[r][c].legend(fontsize=legend_fontsize, labelspacing=0.2)


# ------------------------------------------------------------------------
# Subplot 2: TRAIN ACCURACY (IID)
# ------------------------------------------------------------------------
split        = 'train'
icl_sampling = 'iid'
sample_ratio = '[0.200, 0.200, 0.200, 0.200, 0.200]'
y_value      = 'acc_z_8'
x_label      = x_label_name
y_label      = 'Acc on $z$ (i.i.d.)'
title_str    = 'Training Curves on ID Hypotheses'
r, c = 1, 0

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

axs[r][c].set_xlabel(x_label, fontsize=x_label_fontsize)
axs[r][c].set_ylabel(y_label, fontsize=y_label_fontsize)
#axs[r][c].set_title(title_str, fontsize=title_fontsize)

axs[r][c].set_xlim(0, 768)
axs[r][c].set_ylim(0.00, 1.00)

axs[r][c].set_xticks([0, 256, 512, 768])
axs[r][c].set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])

axs[r][c].tick_params(axis='x', labelsize=tick_fontsize)
axs[r][c].tick_params(axis='y', labelsize=tick_fontsize)

#axs[r][c].legend(fontsize=legend_fontsize)

# ------------------------------------------------------------------------
# Subplot 3: TEST ACCURACY (IID)
# ------------------------------------------------------------------------
split        = 'testI'
icl_sampling = 'iid'
sample_ratio = '[0.200, 0.200, 0.200, 0.200, 0.200]'
y_value      = 'acc_z_8'
x_label      = x_label_name
y_label      = 'Acc on $z$ (i.i.d.)'
title_str    = 'Testing Curves on ID Hypotheses'
r, c = 1, 1

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

axs[r][c].set_xlabel(x_label, fontsize=x_label_fontsize)
axs[r][c].set_ylabel(y_label, fontsize=y_label_fontsize)
#axs[r][c].set_title(title_str, fontsize=title_fontsize)

axs[r][c].set_xlim(0, 768)
axs[r][c].set_ylim(0.00, 1.00)

axs[r][c].set_xticks([0, 256, 512, 768])
axs[r][c].set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])

axs[r][c].tick_params(axis='x', labelsize=tick_fontsize)
axs[r][c].tick_params(axis='y', labelsize=tick_fontsize)

#axs[r][c].legend(fontsize=legend_fontsize)

# ------------------------------------------------------------------------
# Subplot 4: TEST ACCURACY (OPTIMAL / TEACHING SAMPLES)
# ------------------------------------------------------------------------
split        = 'testI'
icl_sampling = 'optimal'
sample_ratio = '[None]'
y_value      = 'acc_z_8'
x_label      = x_label_name
y_label      = 'Acc on $z$ (Opt-T)'
title_str    = 'Testing Curves on ID Hypotheses'
r, c = 0, 1

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

#axs[r][c].set_xlabel(x_label, fontsize=x_label_fontsize)
axs[r][c].set_ylabel(y_label, fontsize=y_label_fontsize)
axs[r][c].set_title(title_str, fontsize=title_fontsize)

axs[r][c].set_xlim(0, 768)
axs[r][c].set_ylim(0.00, 1.00)

axs[r][c].set_xticks([0, 256, 512, 768])
axs[r][c].set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])

axs[r][c].tick_params(axis='x', labelsize=tick_fontsize)
axs[r][c].tick_params(axis='y', labelsize=tick_fontsize)

#axs[r][c].legend(fontsize=legend_fontsize)

# ------------------------------------------------------------------------
# Subplot 3: TEST ACCURACY (IID)
# ------------------------------------------------------------------------
split        = 'testO'
icl_sampling = 'iid'
sample_ratio = '[0.200, 0.200, 0.200, 0.200, 0.200]'
y_value      = 'acc_z_8'
x_label      = x_label_name
y_label      = 'Acc on $z$ (i.i.d.)'
title_str    = 'Testing Curves on OOD Hypotheses'
r, c = 1, 2

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

axs[r][c].set_xlabel(x_label, fontsize=x_label_fontsize)
axs[r][c].set_ylabel(y_label, fontsize=y_label_fontsize)
#axs[r][c].set_title(title_str, fontsize=title_fontsize)

axs[r][c].set_xlim(0, 768)
axs[r][c].set_ylim(0.00, 1.00)

axs[r][c].set_xticks([0, 256, 512, 768])
axs[r][c].set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])

axs[r][c].tick_params(axis='x', labelsize=tick_fontsize)
axs[r][c].tick_params(axis='y', labelsize=tick_fontsize)

#axs[r][c].legend(fontsize=legend_fontsize)

# ------------------------------------------------------------------------
# Subplot 4: TEST ACCURACY (OPTIMAL / TEACHING SAMPLES)
# ------------------------------------------------------------------------
split        = 'testO'
icl_sampling = 'optimal'
sample_ratio = '[None]'
y_value      = 'acc_z_8'
x_label      = x_label_name
y_label      = 'Acc on $z$ (Opt-T)'
title_str    = 'Testing Curves on OOD Hypotheses'
r, c = 0, 2

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

#axs[r][c].set_xlabel(x_label, fontsize=x_label_fontsize)
axs[r][c].set_ylabel(y_label, fontsize=y_label_fontsize)
axs[r][c].set_title(title_str, fontsize=title_fontsize)

axs[r][c].set_xlim(0, 768)
axs[r][c].set_ylim(0.00, 1.00)

axs[r][c].set_xticks([0, 256, 512, 768])
axs[r][c].set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])

axs[r][c].tick_params(axis='x', labelsize=tick_fontsize)
axs[r][c].tick_params(axis='y', labelsize=tick_fontsize)

#axs[r][c].legend(fontsize=legend_fontsize)

# ------------------------------------------------------------------------
# 3. Finalize and Save the Single Figure
# ------------------------------------------------------------------------
plt.tight_layout()  # helps prevent label overlap
plt.savefig(f"FIG_{setup}/multiple_models_for_IO_2x3.pdf")
#plt.savefig(f"FIG_{setup}/multiple_curves_for_IO_2x3.png")
plt.show()
plt.close()