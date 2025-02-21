<h1 align="center"> <p> In-Context Learning with Hypothesis-Class Guidance </p></h1>
<h4 align="center">
    <p><a href="https://myhakureimu.github.io/" target="_blank">Ziqian Lin</a>, <a href="https://skbharti.github.io/" target="_blank">Shubham Kumar Bharti</a>, <a href="https://kangwooklee.com/aboutme/" target="_blank">Kangwook Lee</a></p>
    <p>UW-Madison</p>
    </h4>

**Paper Link**: ToBeRevised [arxiv.org/abs/2402.18819](https://arxiv.org/abs/2402.18819)

Recent research has investigated the underlying mechanisms of in-context learning (ICL) both theoretically and empirically, often using data generated from simple function classes.
However, the existing work often focuses on the sequence consists solely of labeled examples, while in practice, labeled examples are typically accompanied by an *instruction*, providing some side information about the task. 
In this work, we propose **ICL with hypothesis-class guidance (ICL-HCG)**, a novel synthetic data model for ICL where the input context consists of the literal description of a (finite) hypothesis class ℋ and $(x,y)$ pairs from a hypothesis chosen from ℋ.
Under our framework ICL-HCG, we conduct extensive experiments to explore: 
(i) varied generalization ability to new hypothesis classes; 
(ii) different model architectures;
(iii) sample complexity;
(iv) in-context data imbalance;
(v) the role of instruction; and
(vi) the effect of pretraining hypothesis diversity.
As a result, we show that 
(a) Transformers can successfully learn ICL-HCG and generalize to unseen hypotheses and unseen hypothesis classes, and (b) compared with ICL without instruction, ICL-HCG achieves significantly higher accuracy, demonstrating the role of instructions. 

# Experiments (total ~510 Geforce 4090 hours)
The following sections give guidance for reproducing all the experiments in the paper.

## Environment
### System
Ubuntu 22.04.3 LTS + Geforce 4090 

### Package
see the install_package.txt to install everything for our experiments from a plain reserved server. Each training consume ~10 hours on 4090, so we recommend to reserve 4xGeforce 4090 if you are interested in reproduce our results.

## Fig. 5&6 in Sec. 4.2. Four Types of Generalization
### Step 1: Go to the Folder ICL-HCG/
```bash
cd ICL-HCG
```
### Step 2: Training Models (8 file * 1 run/file * 10+h/run / 4 = 20+ GPU hours)
```bash
python EXP_FourGeneration/IO_0.py
python EXP_FourGeneration/IO_1.py
python EXP_FourGeneration/IO_2.py
python EXP_FourGeneration/IO_3.py
python EXP_FourGeneration/IOS_0.py
python EXP_FourGeneration/IOS_1.py
python EXP_FourGeneration/IOS_2.py
python EXP_FourGeneration/IOS_3.py
```
You can parallel run them via running them in different tmux windows.
Arange 4 python files in one 4090 GPU is practically a good idea and consuming all calculation power.
If you are interested in how the hyperparameters are chosen, please read these python file :D.
### Step 3: Visualize Results
Then run
```bash
cd plot/EXP_FourGeneralization
python IO_separate.py
python IOS_separate.py
```
In order to get corresponding Fig. 14 & 15.
```bash
python IO_together.py
python IOS_together.py
```

## Fig. 7&8 in Sec. 4.3. Model Architecture Comparisons
### Step 1: Go to the Folder ICL-HCG/
```bash
cd ICL-HCG
```
### Step 2: Training Models (8 file * 3 run/file * 7 h/run / 4 = 40+ GPU hours)
```bash
python EXP_FourGeneration/IO_other_0.py
python EXP_FourGeneration/IO_other_1.py
python EXP_FourGeneration/IO_other_2.py
python EXP_FourGeneration/IO_other_3.py
python EXP_FourGeneration/IOS_other_0.py
python EXP_FourGeneration/IOS_other_1.py
python EXP_FourGeneration/IOS_other_2.py
python EXP_FourGeneration/IOS_other_3.py
```
You can parallel run them via running them in different tmux windows.
Arange 4 python files in one 4090 GPU is practically a good idea and consuming all calculation power.
If you are interested in how the hyperparameters are chosen, please read these python file :D.
### Step 3: Visualize Results
Then run
```bash
cd plot/EXP_MODEL
python IO_separate.py
python IOS_separate.py
```
In order to get corresponding Fig. 16 & 17.
```bash
python IO_together.py
python IOS_together.py
```

## Fig. 9 in Sec. 4.4. Effect of Training Hypothesis Class Count
### Step 1: Go to the Folder ICL-HCG/
```bash
cd ICL-HCG
```
### Step 2: Training Models (32 file * 2~6 run/file * 10-h/run / 4 = 320- GPU hours, other models are faster)
```bash
python EXP_NUMTRAIN/IO_t_A0.py
...
python EXP_NUMTRAIN/IO_t_A7.py
python EXP_NUMTRAIN/IO_t_B0.py
...
python EXP_NUMTRAIN/IO_t_B7.py
python EXP_NUMTRAIN/IO_other_A0.py
...
python EXP_NUMTRAIN/IO_other_A7.py
python EXP_NUMTRAIN/IO_other_B0.py
...
python EXP_NUMTRAIN/IO_other_B7.py
```
You can parallel run them via running them in different tmux windows.
Arange 4 python files in one 4090 GPU is practically a good idea and consuming all calculation power. (So there are 32 python files, each contains 2 runs on Transformer and 6 runs on other models, 4 or 8 Geforce 4090 are recommended to finish them in 3-2 days.)
If you are interested in how the hyperparameters are chosen, please read these python file :D.
### Step 3: Visualize Results
Then run
```bash
cd plot/EXP_NUMTRAIN
python IO.py
```
In order to get corresponding Fig. 18.
```bash
python IO_together.py
```

## Fig. 10 in Sec. 4.5. Effect of Imbalanced In-Context Samples
### Step 1: Go to the Folder ICL-HCG/
```bash
cd ICL-HCG
```
### Step 2: Training Models (12 file * 1 run/file * 10+h/run / 4 = 30+ GPU hours)
```bash
python EXP_IMBALANCE/IO_1_0.py
python EXP_IMBALANCE/IO_1_1.py
python EXP_IMBALANCE/IO_1_2.py
python EXP_IMBALANCE/IO_1_3.py
python EXP_IMBALANCE/IO_4_0.py
python EXP_IMBALANCE/IO_4_1.py
python EXP_IMBALANCE/IO_4_2.py
python EXP_IMBALANCE/IO_4_3.py
python EXP_IMBALANCE/IO_9_0.py
python EXP_IMBALANCE/IO_9_1.py
python EXP_IMBALANCE/IO_9_2.py
python EXP_IMBALANCE/IO_9_3.py
```
You can parallel run them via running them in different tmux windows.
Arange 4 python files in one 4090 GPU is practically a good idea and consuming all calculation power.
If you are interested in how the hyperparameters are chosen, please read these python file :D.
### Step 3: Visualize Results
Then run
```bash
cd plot/EXP_IMBALANCE
python IO.py
```

## Fig. 11 in Sec. 4.6. The Benefit of Hypothesis Prefix
### Step 1: Go to the Folder ICL-HCG/
```bash
cd ICL-HCG
```
### Step 2: Training Models (8 file * 1 run/file * 10+h/run / 4 = 20+ GPU hours)
```bash
python EXP_ICL/IO_0h_0.py
python EXP_ICL/IO_0h_1.py
python EXP_ICL/IO_0h_2.py
python EXP_ICL/IO_0h_3.py
python EXP_ICL/IO_1h_0.py
python EXP_ICL/IO_1h_1.py
python EXP_ICL/IO_1h_2.py
python EXP_ICL/IO_1h_3.py
```
You can parallel run them via running them in different tmux windows.
Arange 4 python files in one 4090 GPU is practically a good idea and consuming all calculation power.
If you are interested in how the hyperparameters are chosen, please read these python file :D.
### Step 3: Visualize Results
Then run
```bash
cd plot/EXP_ICL
python IO.py
```

## Fig. 12 in Sec. 4.7. The Benefit of Hypothesis Prefix
### Step 1: Go to the Folder ICL-HCG/
```bash
cd ICL-HCG
```
### Step 2: Training Models (16 file * 2 run/file * 10+h/run / 4 = 80+ GPU hours)
```bash
python EXP_Diversity/IO_1h8_0.py
python EXP_Diversity/IO_1h8_1.py
python EXP_Diversity/IO_1h8_2.py
python EXP_Diversity/IO_1h8_3.py
python EXP_Diversity/IO_1h16_0.py
python EXP_Diversity/IO_1h16_1.py
python EXP_Diversity/IO_1h16_2.py
python EXP_Diversity/IO_1h16_3.py
python EXP_Diversity/IO_1h24_0.py
python EXP_Diversity/IO_1h24_1.py
python EXP_Diversity/IO_1h24_2.py
python EXP_Diversity/IO_1h24_3.py
python EXP_Diversity/IO_1h32_0.py
python EXP_Diversity/IO_1h32_1.py
python EXP_Diversity/IO_1h32_2.py
python EXP_Diversity/IO_1h32_3.py
```
You can parallel run them via running them in different tmux windows.
Arange 4 python files in one 4090 GPU is practically a good idea and consuming all calculation power.
If you are interested in how the hyperparameters are chosen, please read these python file :D.
### Step 3: Visualize Results
Then run
```bash
cd plot/EXP_Diversity
python IO.py
```
