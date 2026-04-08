# Probabilistic Prediction of Neural Dynamics via Autoregressive Flow Matching

Code for the paper: **Probabilistic Prediction of Neural Dynamics via Autoregressive Flow Matching**  
Nicole Rogalla, Yuzhen Qin, Mario Senden, Ahmed El-Gazzar, Marcel van Gerven

This repository contains the implementation of a probabilistic generative forecasting framework for modeling fMRI neural dynamics using autoregressive flow matching (AFM). The model predicts future parcel-wise BOLD activity conditioned on past neural dynamics and multimodal sensory input.

---

## Abstract
Forecasting neural activity in response to naturalistic stimuli remains a key challenge for understanding brain dynamics and enabling downstream neurotechnological applications. Here, we introduce a generative forecasting framework for modeling neural dynamics based on autoregressive flow matching (AFM). Building on recent advances in transport-based generative modeling, our approach probabilistically predicts neural responses at scale from multimodal sensory input. Specifically, we learn the conditional distribution of future neural activity given past neural dynamics and concurrent sensory input, explicitly modeling neural activity as a temporally evolving process in which future states depend on recent neural history. 
We evaluate our framework on the Algonauts project 2025 challenge functional magnetic resonance imaging dataset using subject-specific models. AFM significantly outperforms both a non-autoregressive flow-matching baseline and the official challenge general linear model baseline in predicting short-term parcel-wise blood oxygenation level-dependent (BOLD) activity, demonstrating improved generalization and widespread cortical prediction performance. Ablation analyses show that access to past BOLD dynamics is a dominant driver of performance, while autoregressive factorization yields consistent, modest gains under short-horizon, context-rich conditions. Together, these findings position autoregressive flow-based generative modeling as an effective approach for short-term probabilistic forecasting of neural dynamics with promising applications in closed-loop neurotechnology.

---
## Dataset
This project uses the Algonauts 2025 challenge dataset based on the Courtois NeuroMod project:
https://github.com/courtois-neuromod/algonauts_2025.competitors
Please follow the their instructions to obtain access to the dataset.

---

## Forecasting neural dynamics

```bash
pip install -r requirements.txt 

# training and sampling AFM individual models
python src/main.py --framework afm --modeltype gru --area None --modality "all" --use_optimized
```
---
## Reproducing the figures

The notebooks reproduce the figures and analyses reported in the paper.
They require:
- The Algonauts 2025 dataset
- Precomputed model predictions
- Installation of FreeSurfer 7.3.2 for flatmaps