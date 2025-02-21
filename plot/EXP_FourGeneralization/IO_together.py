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

setup = 'IOHypothesis_h+xy+z'
step_name      = 'Step'
x_label_name   = 'Epoch'

sample_ratio = '[0.200, 0.200, 0.200, 0.200, 0.200]'

folder_name = f'FIG_{setup}'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(8*3, 6*2),sharex=True)  # wide figure

# ------------------------------------------------------------------------
# Subplot 0-0: TRAIN LOSS (IID)
# ------------------------------------------------------------------------
split        = 'train'
icl_sampling = 'iid'
y_value      = 'loss__8'
x_label      = x_label_name
y_label      = 'Cross Entropy Loss on $s$'
title_str    = 'Training Curves on ID Hypotheses'
r, c = 0, 0
for idx in range(1, num_runs + 1):
    x_list = [x for x in np.arange(512//2*3+1) if x % 32 == 0][1:]
    y_list = []
    seed = 2023 + idx - 1
    for x in x_list:
        with open(f'../../saved4plot/FourGeneralization/{setup}/model={model} seed={seed}/{x}.pkl', 'rb') as f:
            data = pickle.load(f)
            y = data[f"{split}-{icl_sampling}{sample_ratio}/{y_value}"].item()
            y_list.append(y)
            
    axs[r][c].plot(
        x_list,
        y_list,
        label=labels[idx - 1],
        linewidth=LW,
        zorder=10
    )

#axs[r][c].set_xlabel(x_label, fontsize=x_label_fontsize)
axs[r][c].set_ylabel(y_label, fontsize=y_label_fontsize)
axs[r][c].set_title(title_str, fontsize=title_fontsize)

axs[r][c].set_xlim(0, 768)
axs[r][c].set_ylim(0.25, 0.29)

axs[r][c].set_xticks([0, 256, 512, 768])
axs[r][c].set_yticks([0.25, 0.26, 0.27, 0.28, 0.29])

axs[r][c].tick_params(axis='x', labelsize=tick_fontsize)
axs[r][c].tick_params(axis='y', labelsize=tick_fontsize)

axs[r][c].legend(fontsize=legend_fontsize, labelspacing=0.2)


# ------------------------------------------------------------------------
# Subplot 2: TRAIN ACCURACY (IID)
# ------------------------------------------------------------------------
split        = 'train'
icl_sampling = 'iid'
y_value      = 'acc_z_8'
x_label      = x_label_name
y_label      = 'Acc on $z$ (i.i.d.)'
title_str    = 'Training Curves on ID Hypotheses'
r, c = 1, 0
for idx in range(1, num_runs + 1):
    x_list = [x for x in np.arange(512//2*3+1) if x % 32 == 0][1:]
    y_list = []
    seed = 2023 + idx - 1
    for x in x_list:
        with open(f'../../saved4plot/FourGeneralization/{setup}/model={model} seed={seed}/{x}.pkl', 'rb') as f:
            data = pickle.load(f)
            y = data[f"{split}-{icl_sampling}{sample_ratio}/{y_value}"].item()
            y_list.append(y)
            
    axs[r][c].plot(
        x_list,
        y_list,
        label=labels[idx - 1],
        linewidth=LW,
        zorder=10
    )

axs[r][c].set_xlabel(x_label, fontsize=x_label_fontsize)
axs[r][c].set_ylabel(y_label, fontsize=y_label_fontsize)
#axs[r][c].set_title(title_str, fontsize=title_fontsize)

axs[r][c].set_xlim(0, 512)
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
y_value      = 'acc_z_8'
x_label      = x_label_name
y_label      = 'Acc on $z$ (i.i.d.)'
title_str    = 'Testing Curves on ID Hypotheses'
r, c = 1, 1
for idx in range(1, num_runs + 1):
    x_list = [x for x in np.arange(512//2*3+1) if x % 32 == 0]
    y_list = []
    seed = 2023 + idx - 1
    for x in x_list:
        with open(f'../../saved4plot/FourGeneralization/{setup}/model={model} seed={seed}/{x}.pkl', 'rb') as f:
            data = pickle.load(f)
            y = data[f"{split}-{icl_sampling}{sample_ratio}/{y_value}"].item()
            y_list.append(y)
            
    axs[r][c].plot(
        x_list,
        y_list,
        label=labels[idx - 1],
        linewidth=LW,
        zorder=10
    )

axs[r][c].set_xlabel(x_label, fontsize=x_label_fontsize)
axs[r][c].set_ylabel(y_label, fontsize=y_label_fontsize)
#axs[r][c].set_title(title_str, fontsize=title_fontsize)

axs[r][c].set_xlim(0, 512)
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
y_value      = 'acc_z_8'
x_label      = x_label_name
y_label      = 'Acc on $z$ (Opt-T)'
title_str    = 'Testing Curves on ID Hypotheses'
r, c = 0, 1
for idx in range(1, num_runs + 1):
    x_list = [x for x in np.arange(512//2*3+1) if x % 32 == 0]
    y_list = []
    seed = 2023 + idx - 1
    for x in x_list:
        with open(f'../../saved4plot/FourGeneralization/{setup}/model={model} seed={seed}/{x}.pkl', 'rb') as f:
            data = pickle.load(f)
            y = data[f"{split}-{icl_sampling}[None]/{y_value}"].item()
            y_list.append(y)
            
    axs[r][c].plot(
        x_list,
        y_list,
        label=labels[idx - 1],
        linewidth=LW,
        zorder=10
    )

#axs[r][c].set_xlabel(x_label, fontsize=x_label_fontsize)
axs[r][c].set_ylabel(y_label, fontsize=y_label_fontsize)
axs[r][c].set_title(title_str, fontsize=title_fontsize)

axs[r][c].set_xlim(0, 512)
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
y_value      = 'acc_z_8'
x_label      = x_label_name
y_label      = 'Acc on $z$ (i.i.d.)'
title_str    = 'Testing Curves on OOD Hypotheses'
r, c = 1, 2
for idx in range(1, num_runs + 1):
    x_list = [x for x in np.arange(512//2*3+1) if x % 32 == 0]
    y_list = []
    seed = 2023 + idx - 1
    for x in x_list:
        with open(f'../../saved4plot/FourGeneralization/{setup}/model={model} seed={seed}/{x}.pkl', 'rb') as f:
            data = pickle.load(f)
            y = data[f"{split}-{icl_sampling}{sample_ratio}/{y_value}"].item()
            y_list.append(y)
            
    axs[r][c].plot(
        x_list,
        y_list,
        label=labels[idx - 1],
        linewidth=LW,
        zorder=10
    )

axs[r][c].set_xlabel(x_label, fontsize=x_label_fontsize)
axs[r][c].set_ylabel(y_label, fontsize=y_label_fontsize)
#axs[r][c].set_title(title_str, fontsize=title_fontsize)

axs[r][c].set_xlim(0, 512)
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
y_value      = 'acc_z_8'
x_label      = x_label_name
y_label      = 'Acc on $z$ (Opt-T)'
title_str    = 'Testing Curves on OOD Hypotheses'
r, c = 0, 2
for idx in range(1, num_runs + 1):
    x_list = [x for x in np.arange(512//2*3+1) if x % 32 == 0]
    y_list = []
    seed = 2023 + idx - 1
    for x in x_list:
        with open(f'../../saved4plot/FourGeneralization/{setup}/model={model} seed={seed}/{x}.pkl', 'rb') as f:
            data = pickle.load(f)
            y = data[f"{split}-{icl_sampling}[None]/{y_value}"].item()
            y_list.append(y)
            
    axs[r][c].plot(
        x_list,
        y_list,
        label=labels[idx - 1],
        linewidth=LW,
        zorder=10
    )

#axs[r][c].set_xlabel(x_label, fontsize=x_label_fontsize)
axs[r][c].set_ylabel(y_label, fontsize=y_label_fontsize)
axs[r][c].set_title(title_str, fontsize=title_fontsize)

axs[r][c].set_xlim(0, 512)
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
plt.savefig(f"FIG_{setup}/multiple_curves_for_IO_2x3.pdf")
#plt.savefig(f"FIG_{setup}/multiple_curves_for_IO_2x3.png")
plt.show()
plt.close()