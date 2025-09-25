# Investigating Large Language Model Hallucinations on Known Facts


## Abstract
Large language models (LLMs) have
demonstrated strong performance in answering factoid
questions but remain prone to hallucinations, producing
incorrect outputs despite possessing the correct knowledge. This
study investigates the internal inference dynamics underlying
these errors across diverse models (LLaMA3-8B, GPT-2,
Pythia-70M, OPT-125M). By tracking output token
probabilities across layers using Logit and Tuned Lens, we
identify distinct dynamic patterns differentiating correct recalls
from hallucinations. Correct recalls typically show a sharp
probability increase in later layers, while hallucinations exhibit
erratic or premature gains. Leveraging these dynamics as
features, Support Vector Machine (SVM) classifiers successfully
detect hallucinations with high accuracy (80-90%) across all
tested models, including smaller architectures. These findings
demonstrate that internal inference dynamics provide a robust,
generalizable signal for hallucination detection, enabling
monitoring based solely on intermediate model states without
external labels.

## Work done
• Analyzed hallucination patterns in LLaMA3-8B, GPT-2, Pythia-70M, and OPT-125M using 30K+ factual triplet-based queries adapted from the COUNTERFACT dataset.
• Applied Logit Lens and Tuned Lens to trace token probability trajectories across transformer layers, revealing distinct inference dynamics between correct recalls (late-layer spikes) and hallucinations (unstable or premature gains).
• Extracted dynamic features such as probability trends, token dominance, and rate of change across layers to model internal knowledge recall mechanisms.
• Trained a Support Vector Machine (SVM) classifier, achieving 80–90% hallucination detection accuracy across all models, with smaller models (OPT-125M, Pythia-70M) performing particularly well.
## Key Findings
• Hallucinations often result from ineffective inference dynamics rather than missing knowledge, and combined logit + tuned features yield more robust and generalizable detection signals.
