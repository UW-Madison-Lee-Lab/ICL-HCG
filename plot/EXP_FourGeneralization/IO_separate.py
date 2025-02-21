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
            
    for idx in range(1, num_runs + 1):
        seed = 2023 + idx - 1
        x_list = [x for x in np.arange(512//2*3+1) if x % 32 == 0]
        y_list = []
        for x in x_list:
            with open(f'../../saved4plot/FourGeneralization/{setup}/model={model} seed={seed}/{x}.pkl', 'rb') as f:
                data = pickle.load(f)
                y = data[f"{split}-{icl_sampling}[None]/{y_value}"].item()
                y_list.append(y)
                
        plt.plot(
            x_list,
            y_list,
            label=labels[idx - 1],
            linewidth=LW,
            zorder=10
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
    plt.savefig(f"FIG_{setup}/multiple_curves_for_{key}_1x1.pdf")
    #plt.savefig(f"FIG_{setup}/multiple_curves_for_{key}_1x1.png")
    plt.show()
    plt.close()