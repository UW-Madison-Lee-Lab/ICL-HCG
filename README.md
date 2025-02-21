<h1 align="center"> <p> In-Context Learning with Hypothesis-Class Guidance </p></h1>
<h4 align="center">
    <p><a href="https://myhakureimu.github.io/" target="_blank">Ziqian Lin</a>, <a href="https://skbharti.github.io/" target="_blank">Shubham Kumar Bharti</a>, <a href="https://kangwooklee.com/aboutme/" target="_blank">Kangwook Lee</a></p>
    <p>UW-Madison</p>
    </h4>

**Paper Link**: [arxiv.org/abs/2402.18819](https://arxiv.org/abs/2402.18819)

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

# Experiments
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
### Step 2: Training Models
```bash
python FourGeneration/IO_0.py
python FourGeneration/IO_1.py
python FourGeneration/IO_2.py
python FourGeneration/IO_3.py
python FourGeneration/IOS_0.py
python FourGeneration/IOS_1.py
python FourGeneration/IOS_2.py
python FourGeneration/IOS_3.py
```
You can parallel run them via running them in different tmux windows.
Arange 4 runs in one 4090 GPU is pratically a good idea and consuming all calculation power.
If you are interested in how the hyperparameters are chosen, please read these python file :D.
### Step 3: Visualize Results
Then run
```bash
cd plot/EXP_FourGeneralization
IO_separate.py
IOS_separate.py
```
In order to get corresponding Fig. 14 & 15.
```bash
IO_together.py
IOS_together.py
```
### Figure 5
#### Step 1: Go to the Folder
```bash
cd NumericalComputation/Figure5/
```
#### Step 2 (Method 1): Get Results from Scratch
```bash
python 5.1.2_Preprocess.py
```
One can reduce the sample size "K = 80000" for the Monte Carlo simulation in the code to accelerate the process, though this will likely result in increased variance.
#### Step 2 (Method 2): Download Results from Dropbox
Download and unzip the corresponding .zip file from [Dropbox link](https://www.dropbox.com/scl/fo/q0rj5eyfd9wasatbnpy7r/h?rlkey=epjq87hvf3br3ljqa6a1g50bn&dl=0).
#### Step 3: Visualize Results
Then run
```bash
python 5.1.2_Visualize.py
```
to get Figure5.pdf.

### Figure 6
#### Step 1: Go to the Folder
```bash
cd NumericalComputation/Figure5/
```
#### Step 2 (Method 1): Get Results from Scratch
```bash
python EarlyAscent_Preprocess.py
```
One can reduce the sample size "K = 10000" for the Monte Carlo simulation in the code to accelerate the process, though this will likely result in increased variance.
**Note**: The code takes a long time to run since it loops through these parameters:
d_list = \[1,2,3,5,8\]
and
demon_list = \[0,1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072\].

#### Step 2 (Method 2): Download Results from Dropbox
Download and unzip the corresponding .zip file from [Dropbox link](https://www.dropbox.com/scl/fo/q0rj5eyfd9wasatbnpy7r/h?rlkey=epjq87hvf3br3ljqa6a1g50bn&dl=0).
#### Step 3: Visualize Results
Then run
```bash
python EarlyAscent_Visualize.py
```
to get Figure6.pdf.

## RealWorld LLM Experiment
### Table 1 (running with GPT-4 2023/11/20)
#### Step 1: Go to the Folder
```bash
cd RealWorldLLMExperiment/Table1/
```
#### Step 2: Register Your Openai key
```bash
vi call_openai.py
```
Replace the string "yourkey" in the code with your Openai key.
#### Step 3: Get Results from Scratch
For k (for instance k=4) in-context examples, run
```bash
python Ushape.py --k 4
```

### Figure 8
Note: In the following codes, the inferences of llama2, mistral, and mixtral are based on [vllm](https://docs.vllm.ai/en/latest/). One will need at least 4xA100 to run the biggest models, including mixtral and llama-2-70b-hf.
#### Step 1: Go to the Folder
```bash
cd RealWorldLLMExperiment/Figure8/
```
#### Step 2: Register Your Openai key
```bash
vi call_openai.py
```
Replace the string "yourkey" in the code with your Openai key.
#### Step 3 (Method 1): Get Results from Scratch
```bash
python test_gpt4.py
python test_llama-2-13b-hf.py
python test_llama-2-70b-hf.py
python test_mistral.py
python test_mixtral.py
```
#### Step 3 (Method 2): Download Results from Dropbox
Download and unzip the corresponding .zip file from [Dropbox link](https://www.dropbox.com/scl/fo/q0rj5eyfd9wasatbnpy7r/h?rlkey=epjq87hvf3br3ljqa6a1g50bn&dl=0).
#### Step 4: Visualize Results
After step 3, run:
```
python ZeroICL.py
```

## Transformer Experiment
The following code can be run on a single 4090 GPU.
#### Step 1: Go to the Folder
```bash
cd TransformerExperiment/
```
### Figure 9
#### Step 2 (Method 1): Get Results from Scratch
```bash
python TS_Regular4_delta_run.py
```
#### Step 2 (Method 2): Download Results from Dropbox
Download and unzip the corresponding .zip file from [Dropbox link](https://www.dropbox.com/scl/fo/q0rj5eyfd9wasatbnpy7r/h?rlkey=epjq87hvf3br3ljqa6a1g50bn&dl=0).
#### Step 3: Visualize Results
```bash
python TS_Regular4_delta_visual.py
```
### Figure 10
#### Step 2 (Method 1): Get Results from Scratch
```bash
python TS_RegularM_run.py
```
#### Step 2 (Method 2): Download Results from Dropbox
Download and unzip the corresponding .zip file from [Dropbox link](https://www.dropbox.com/scl/fo/q0rj5eyfd9wasatbnpy7r/h?rlkey=epjq87hvf3br3ljqa6a1g50bn&dl=0).
#### Step 3: Visualize Results
```bash
python TS_RegularM_visual.py
```
### Figure 11
#### Step 2 (Method 1): Get Results from Scratch
```bash
python TS_D_d_run.py
```
#### Step 2 (Method 2): Download Results from Dropbox
Download and unzip the corresponding .zip file from [Dropbox link](https://www.dropbox.com/scl/fo/q0rj5eyfd9wasatbnpy7r/h?rlkey=epjq87hvf3br3ljqa6a1g50bn&dl=0).
#### Step 3: Visualize Results
```bash
python TS_D_d_visual.py
```
