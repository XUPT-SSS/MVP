# Mind the Few: Semantic Prompt-Guided Meta-Learning for Vulnerability Type Detection
Vulnerability type detection in source code is essential for securing modern software systems but remains challenging in practice due to limited labeled data for rare or emerging vulnerability types and long-tailed class distributions that cause severe data imbalance. These factors often result in few-shot learning scenarios, where models must generalize from only a few examples per class. Existing deep learning (DL) approaches rely heavily on large, balanced datasets and struggle in such settings. To address this limitation, we propose MVP, a Meta-learning framework for few-shot Vulnerability type detection that guided by semantic Prompts. MVP incorporates CWE vulnerability descriptions into code token embeddings to steer feature extraction, capturing type-specific semantics from limited code samples. Trained with episodic meta-learning and equipped with a prototypical network,MVP enables effective detection across imbalanced and underrepresented vulnerability types. Extensive experiments show that MVP outperforms state-of-the-art baselines in few-shot settings and maintains stable performance on imbalanced and long-tailed distributions, demonstrating strong generalization to rare and emerging vulnerabilities.

# Design of MVP
<div align="center">
  <img src="https://github.com/XUPT-SSS/MVP/blob/main/overview_1.jpg">
</div>

# Requirements
torch==2.0.1
torchvision==0.15.2
tokenizers==0.19.1
transformers==4.40.2
tree_sitter==0.20.1
numpy==1.26.0
scikit-learn==1.4.2
pandas==2.2.2

# Datasets
The dataset can be downloaded at:https://drive.google.com/drive/folders/1lbedAH64v7hlHGeIWJSGfVYkhFb-T5pA?usp=drive_link
# Source
## Step1:Code normalization
Normalize the raw source code to remove noise such as comments, inconsistent indentation, and redundant symbols.
```
cd preprocess
python normalization.py
```
## Step2:Pretrain
Pretrain the model on normalized source code to learn general-purpose code representations.
```
cd ..
python pretrain.py
```
## Step3:Description embedding
Extract semantic embeddings for CWE vulnerability descriptions.
These embeddings act as semantic prompts that guide the feature extraction process during meta-learning.
```
python cwe_desc.py
```
## Step4:Meta-training and Meta-testing
Perform episodic meta-training and meta-testing under few-shot settings.
During training, the model learns to rapidly adapt to new vulnerability types; during testing, it is evaluated on unseen or rare categories.
```
python meta_train.py
```
