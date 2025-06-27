# Mind the Few: Semantic Prompt-Guided Meta-Learning for Vulnerability Type Detection
Vulnerability type detection in source code is essential for securing modern software systems but remains challenging in practice due to limited labeled data for rare or emerging vulnerability types and long-tailed class distributions that cause severe data imbalance. These factors often result in few-shot learning scenarios, where models must generalize from only a few examples per class. Existing deep learning (DL) approaches rely heavily on large, balanced datasets and struggle in such settings. To address this limitation, we propose MVP, a Meta-learning framework for few-shot Vulnerability type detection that guided by semantic Prompts. MVP incorporates CWE vulnerability descriptions into code token embeddings to steer feature extraction, capturing type-specific semantics from limited code samples. Trained with episodic meta-learning and equipped with a prototypical network,MVP enables effective detection across imbalanced and underrepresented vulnerability types. Extensive experiments show that MVP outperforms state-of-the-art baselines in few-shot settings and maintains stable performance on imbalanced and long-tailed distributions, demonstrating strong generalization to rare and emerging vulnerabilities.

# Design of MVP
<div align="center">
  <img src="https://github.com/XUPT-SSS/MVP/blob/main/overview_1.jpg">
</div>

# Datasets
The dataset can be downloaded at:
# Source
## Step1:Code normalization
```
cd preprocess
python normalization.py
```
## Step2:Pretrain
```
cd ..
python pretrain.py
```
## Step3:Description embedding
```
python cwe_desc.py
```
## Step4:Meta-train
```
python meta_train.py
```
