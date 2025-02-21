import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pickle
import os
import numpy as np
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
LW = 3  # thick lines

seeds = [
    2023,
    2024,
    2025,
    2026
]

sparsities = {
    8: '8',
    16: '16',
    24: '24',
    32: '32',
    48: '48',
}

colors = [
    "#b3cde0",  # lightest
    "#6497b1",
    "#005b96",
    "#03396c"   # darkest
]

linestyles = [
    ":",   # dotted
    "-.",  # dash-dot
    "--",  # dashed
    "-"    # solid
]

markers = [
    "o",  # circle
    "^",  # triangle_up
    "d",  # square
    "p"   # diamond
]

markersize = 12

contents = {
    'xy': 'w/o inst,',
    'h+xy': 'w/ inst,'    
}

# Different model architecture
icl_k_list = [i for i in range(12)]

epoch = 768

folder_name = f'FIG'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    
# ------------------------------------------------------------------------
# 2. Create a single figure with 4 subplots (1 row, 4 columns)
# ------------------------------------------------------------------------
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))  # wide figure

fig.subplots_adjust(
    wspace=0.5,  # width space between columns
    hspace=0.3   # height space between rows
)

# ------------------------------------------------------------------------
# Subplot 1: TRAIN LOSS (xy)
# ------------------------------------------------------------------------
# Directory and file naming setup
setup          = 'IOHypothesis_xy'
title_str      = r'Training Curve $\boldsymbol{w/o}$ Instruction'
x_label_name   = 'Position in In-Context Sequence'
x_sticks       = [0,4,8,12]
x_labels       = [f"${i}$" for i in x_sticks]
var_multiplier = 1

r, c = 0, 0
split        = 'train'
icl_sampling = 'iid'
sample_ratio   = '[0.167, 0.167, 0.167, 0.167, 0.167, 0.167]'
y_value      = 'acc_y'
x_label      = x_label_name
y_label      = 'Acc on $y$ (i.i.d.)'


for sparsity, color, linestyle, marker in zip(sparsities.keys(), colors, linestyles, markers):
    ys_list = []
    for icl_k in icl_k_list:
        #df = pd.read_csv(f"{HEAD}/{split}/{Generalization}_{split}-{icl_sampling}_{y_value}_pos={icl_k+1}.csv")
        ys = []
        for seed in seeds:
            #data = df[f"content={content} sparsity={sparsity} seed={seed} - {split}-{icl_sampling}{DP}_icl/pos{icl_k+1}"]
            #ys.append(data.dropna().iloc[-1])
            with open(f'../../saved4plot/Diversity/{setup}/DP={setup[13:]} num={sparsity} seed={seed}/{epoch}.pkl', 'rb') as f:
                data = pickle.load(f)
                y = data[f"{split}-{icl_sampling}{sample_ratio}_icl/pos{icl_k+1}"].item()
            ys.append(y)
        ys_list.append(ys)
        #ys.append(data.dropna().iloc[-1])
        #ys_list.append(ys)
    # Calculate means and variances for data1
    mean_list = [np.mean(group) for group in ys_list]
    max_list  = [np.max (group) for group in ys_list]
    min_list  = [np.min (group) for group in ys_list]
    
    axs[r][c].plot(
        icl_k_list,
        mean_list,
        #color='tab:blue',
        marker=marker,
        markersize=markersize,
        color=color,
        linestyle=linestyle,
        label=fr'$M^{{\text{{train}}}}$={sparsities[sparsity]}' #f'$M^{\text{train}}$={sparsities[sparsity]}'
    )
    # Fill between (mean - var) and (mean + var) to represent "confidence interval"
    axs[r][c].fill_between(
        icl_k_list,
        max_list,
        min_list,
        #color='tab:blue',
        color=color,
        alpha=0.2
    )

axs[r][c].set_xlabel(x_label, fontsize=x_label_fontsize)
axs[r][c].set_ylabel(y_label, fontsize=y_label_fontsize)
axs[r][c].set_title(title_str, fontsize=title_fontsize)

axs[r][c].set_xlim(0, 12)
axs[r][c].set_ylim(0.50, 1.00)

axs[r][c].set_xticks(x_sticks)
axs[r][c].set_xticklabels(x_labels)
axs[r][c].set_yticks([0.50, 0.60, 0.70, 0.80, 0.90, 1.00])

axs[r][c].tick_params(axis='x', labelsize=tick_fontsize)
axs[r][c].tick_params(axis='y', labelsize=tick_fontsize)



# ------------------------------------------------------------------------
# Subplot 2: TRAIN LOSS (hxyz)
# ------------------------------------------------------------------------
# Directory and file naming setup
setup          = 'IOHypothesis_h+xy'
title_str      = r'Training Curve $\boldsymbol{w/}$ Instruction'
x_label_name   = 'Position in In-Context Sequence'
x_sticks       = [0,4,8,12]
x_labels       = [f"${i}$" for i in x_sticks]
var_multiplier = 1

r, c = 0, 1
split        = 'train'
icl_sampling = 'iid'
sample_ratio = '[0.167, 0.167, 0.167, 0.167, 0.167, 0.167]'
y_value      = 'acc_y'
x_label      = x_label_name
y_label      = 'Acc on $y$ (i.i.d.)'

for sparsity, color, linestyle, marker in zip(sparsities.keys(), colors, linestyles, markers):
    ys_list = []
    for icl_k in icl_k_list:
        #df = pd.read_csv(f"{HEAD}/{split}/{Generalization}_{split}-{icl_sampling}_{y_value}_pos={icl_k+1}.csv")
        ys = []
        for seed in seeds:
            #data = df[f"content={content} sparsity={sparsity} seed={seed} - {split}-{icl_sampling}{DP}_icl/pos{icl_k+1}"]
            #ys.append(data.dropna().iloc[-1])
            with open(f'../../saved4plot/Diversity/{setup}/DP={setup[13:]} num={sparsity} seed={seed}/{epoch}.pkl', 'rb') as f:
                data = pickle.load(f)
                y = data[f"{split}-{icl_sampling}{sample_ratio}_icl/pos{icl_k+1}"].item()
            ys.append(y)
        ys_list.append(ys)
        #ys.append(data.dropna().iloc[-1])
        #ys_list.append(ys)
    # Calculate means and variances for data1
    mean_list = [np.mean(group) for group in ys_list]
    max_list  = [np.max (group) for group in ys_list]
    min_list  = [np.min (group) for group in ys_list]
    
    axs[r][c].plot(
        icl_k_list,
        mean_list,
        #color='tab:blue',
        marker=marker,
        markersize=markersize,
        color=color,
        linestyle=linestyle,
        label=fr'$M^{{\text{{train}}}}$={sparsities[sparsity]}' #f'$M^{\text{train}}$={sparsities[sparsity]}'
    )
    # Fill between (mean - var) and (mean + var) to represent "confidence interval"
    axs[r][c].fill_between(
        icl_k_list,
        max_list,
        min_list,
        #color='tab:blue',
        color=color,
        alpha=0.2
    )
    
axs[r][c].set_xlabel(x_label, fontsize=x_label_fontsize)
axs[r][c].set_ylabel(y_label, fontsize=y_label_fontsize)
axs[r][c].set_title(title_str, fontsize=title_fontsize)

axs[r][c].set_xlim(0, 12)
axs[r][c].set_ylim(0.50, 1.00)

axs[r][c].set_xticks(x_sticks)
axs[r][c].set_xticklabels(x_labels)
axs[r][c].set_yticks([0.50, 0.60, 0.70, 0.80, 0.90, 1.00])

axs[r][c].tick_params(axis='x', labelsize=tick_fontsize)
axs[r][c].tick_params(axis='y', labelsize=tick_fontsize)

#axs[r][c].legend(loc='lower right', fontsize=legend_fontsize)
axs[r][c].legend(loc='lower right', fontsize=legend_fontsize)

# ------------------------------------------------------------------------
# Subplot 3: TEST LOSS (xy)
# ------------------------------------------------------------------------
# Directory and file naming setup
setup          = 'IOHypothesis_xy'
title_str      = r'Testing Curve $\boldsymbol{w/o}$ Instruction'
x_label_name   = 'Position in In-Context Sequence'
x_sticks       = [0,4,8,12]
x_labels       = [f"${i}$" for i in x_sticks]
var_multiplier = 1

r, c = 1, 0
split        = 'testO'
icl_sampling = 'iid'
sample_ratio = '[0.167, 0.167, 0.167, 0.167, 0.167, 0.167]'
y_value      = 'acc_y'
x_label      = x_label_name
y_label      = 'Acc on $y$ (i.i.d.)'

for sparsity, color, linestyle, marker in zip(sparsities.keys(), colors, linestyles, markers):
    ys_list = []
    for icl_k in icl_k_list:
        #df = pd.read_csv(f"{HEAD}/{split}/{Generalization}_{split}-{icl_sampling}_{y_value}_pos={icl_k+1}.csv")
        ys = []
        for seed in seeds:
            #data = df[f"content={content} sparsity={sparsity} seed={seed} - {split}-{icl_sampling}{DP}_icl/pos{icl_k+1}"]
            #ys.append(data.dropna().iloc[-1])
            with open(f'../../saved4plot/Diversity/{setup}/DP={setup[13:]} num={sparsity} seed={seed}/{epoch}.pkl', 'rb') as f:
                data = pickle.load(f)
                y = data[f"{split}-{icl_sampling}{sample_ratio}_icl/pos{icl_k+1}"].item()
            ys.append(y)
        ys_list.append(ys)
        #ys.append(data.dropna().iloc[-1])
        #ys_list.append(ys)
    # Calculate means and variances for data1
    mean_list = [np.mean(group) for group in ys_list]
    max_list  = [np.max (group) for group in ys_list]
    min_list  = [np.min (group) for group in ys_list]
    
    axs[r][c].plot(
        icl_k_list,
        mean_list,
        #color='tab:blue',
        marker=marker,
        markersize=markersize,
        color=color,
        linestyle=linestyle,
        label=f'training sparsity={sparsities[sparsity]}'
    )
    # Fill between (mean - var) and (mean + var) to represent "confidence interval"
    axs[r][c].fill_between(
        icl_k_list,
        max_list,
        min_list,
        #color='tab:blue',
        color=color,
        alpha=0.2
    )
    
axs[r][c].set_xlabel(x_label, fontsize=x_label_fontsize)
axs[r][c].set_ylabel(y_label, fontsize=y_label_fontsize)
axs[r][c].set_title(title_str, fontsize=title_fontsize)

axs[r][c].set_xlim(0, 12)
axs[r][c].set_ylim(0.50, 1.00)

axs[r][c].set_xticks(x_sticks)
axs[r][c].set_xticklabels(x_labels)
axs[r][c].set_yticks([0.50, 0.60, 0.70, 0.80, 0.90, 1.00])

axs[r][c].tick_params(axis='x', labelsize=tick_fontsize)
axs[r][c].tick_params(axis='y', labelsize=tick_fontsize)

#axs[r][c].legend(loc='lower right', fontsize=legend_fontsize)


# ------------------------------------------------------------------------
# Subplot 4: TEST LOSS (hxyz)
# ------------------------------------------------------------------------
# Directory and file naming setup
setup          = 'IOHypothesis_h+xy'
title_str      = r'Testing Curve $\boldsymbol{w/}$ Instruction'
x_label_name   = 'Position in In-Context Sequence'
x_sticks       = [0,4,8,12]
x_labels       = [f"${i}$" for i in x_sticks]
var_multiplier = 1

r, c = 1, 1
split        = 'testO'
icl_sampling = 'iid'
sample_ratio = '[0.167, 0.167, 0.167, 0.167, 0.167, 0.167]'
y_value      = 'acc_y'
x_label      = x_label_name
y_label      = 'Acc on $y$ (i.i.d.)'

for sparsity, color, linestyle, marker in zip(sparsities.keys(), colors, linestyles, markers):
    ys_list = []
    for icl_k in icl_k_list:
        #df = pd.read_csv(f"{HEAD}/{split}/{Generalization}_{split}-{icl_sampling}_{y_value}_pos={icl_k+1}.csv")
        ys = []
        for seed in seeds:
            #data = df[f"content={content} sparsity={sparsity} seed={seed} - {split}-{icl_sampling}{DP}_icl/pos{icl_k+1}"]
            #ys.append(data.dropna().iloc[-1])
            with open(f'../../saved4plot/Diversity/{setup}/DP={setup[13:]} num={sparsity} seed={seed}/{epoch}.pkl', 'rb') as f:
                data = pickle.load(f)
                y = data[f"{split}-{icl_sampling}{sample_ratio}_icl/pos{icl_k+1}"].item()
            ys.append(y)
        ys_list.append(ys)
        #ys.append(data.dropna().iloc[-1])
        #ys_list.append(ys)
    # Calculate means and variances for data1
    mean_list = [np.mean(group) for group in ys_list]
    max_list  = [np.max (group) for group in ys_list]
    min_list  = [np.min (group) for group in ys_list]
    
    axs[r][c].plot(
        icl_k_list,
        mean_list,
        #color='tab:blue',
        marker=marker,
        markersize=markersize,
        color=color,
        linestyle=linestyle,
        label=f'training sparsity={sparsities[sparsity]}'
    )
    # Fill between (mean - var) and (mean + var) to represent "confidence interval"
    axs[r][c].fill_between(
        icl_k_list,
        max_list,
        min_list,
        #color='tab:blue',
        color=color,
        alpha=0.2
    )
    
axs[r][c].set_xlabel(x_label, fontsize=x_label_fontsize)
axs[r][c].set_ylabel(y_label, fontsize=y_label_fontsize)
axs[r][c].set_title(title_str, fontsize=title_fontsize)

axs[r][c].set_xlim(0, 12)
axs[r][c].set_ylim(0.50, 1.00)

axs[r][c].set_xticks(x_sticks)
axs[r][c].set_xticklabels(x_labels)
axs[r][c].set_yticks([0.50, 0.60, 0.70, 0.80, 0.90, 1.00])

axs[r][c].tick_params(axis='x', labelsize=tick_fontsize)
axs[r][c].tick_params(axis='y', labelsize=tick_fontsize)

#axs[r][c].legend(loc='lower right', fontsize=legend_fontsize)


# ------------------------------------------------------------------------
# 3. Finalize and Save the Single Figure
# ------------------------------------------------------------------------
plt.tight_layout()  # helps prevent label overlap
plt.savefig(f"FIG/diversity_2x2_combined.pdf")
plt.savefig(f"FIG/diversity_2x2_combined.png")
plt.show()
plt.close()
