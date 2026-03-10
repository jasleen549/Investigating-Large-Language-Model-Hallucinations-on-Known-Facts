# Hallucination Detection in Large Language Models


## Overview
Large Language Models such as LLaMA, GPT, OPT and Pythia often generate responses that appear confident but may contain incorrect or fabricated information. These hallucinations reduce trust in AI systems and present challenges for real-world deployment. This project investigates how internal model signals can be used to detect hallucinated responses in LLM outputs.

## Project Goal
The goal of this project is to build a machine learning system that can classify whether an LLM response is factually correct or hallucinated when answering questions with known factual answers.

## Approach
The detection framework consists of the following steps:
#### 1. Response Generation
Generate answers from multiple LLM architectures.
#### 2. Model Interpretability Analysis
Use Logit Lens and Tuned Lens techniques to inspect token probabilities across transformer layers.
#### 3. Feature Engineering
Extract signals from intermediate model outputs that correlate with hallucinated responses.
#### 4. Classification
Train a Support Vector Machine (SVM) classifier to detect hallucinations.
#### 5. Evaluation
Evaluate performance using: Accuracy, Precision, Recall, F1-score, AUC

## Results
The hallucination detection model achieved:
1. 94% accuracy
2. strong precision and recall across generated responses
3. consistent performance across multiple LLM architectures
These results demonstrate that internal model signals can be leveraged to identify hallucinations before responses reach end users.

## Potential Applications
This system can support the development of trustworthy AI systems, including:
1. AI assistants with reliability monitoring
2. automated knowledge systems with hallucination detection
3. enterprise AI governance frameworks
4. safety layers for production LLM deployments

## Tech Stack
Python, Scikit-learn, Transformer models, Logit Lens / Tuned Lens, Data analysis and evaluation pipelines

## Future Work
1. Evaluate across additional LLM architectures
2. Extend detection to open-ended prompts
3. Integrate detection system into real-time AI workflows
