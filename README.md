# wavlm-rag-student-engagement
Code for WavLM-RAG, a speech-based student engagement classification framework using residual adaptive gating to combine temporal speech representations with global acoustic features.
# WavLM-RAG

## Residual Adaptive Gating of Temporal and Global Acoustic Representations for Speech-Based Student Engagement Classification

Research code accompanying the manuscript by **Sneha Sureddy and I. Jeena Jacob**.

## Overview

WavLM-RAG classifies student engagement from speech into three categories: **High (H)**, **Low (L)**, and **Partially engaged (P)**. The framework combines temporal representations from WavLM with global acoustic features through a bounded adaptive residual gate.

Here, **RAG means Residual Adaptive Gating**.

## Method

1. **Audio preparation:** Convert recordings to mono and resample to 16 kHz. The WavLM branch uses up to 8 seconds per recording, with center cropping for longer recordings. Global features are extracted from the full resampled waveform.
2. **Temporal branch:** Extract WavLM-Base-Plus representations and apply attentive statistics pooling. Only the final two encoder layers are fine-tuned.
3. **Global acoustic branch:** Combine 91 handcrafted acoustic features with 256-dimensional PCA-reduced HuBERT representations, producing a 347-dimensional feature vector.
4. **Residual fusion:** Project both branches to 192-dimensional embeddings. A gate conditioned on the branch embeddings, recording-quality descriptors, and prediction uncertainty controls the global branch contribution, with a maximum residual coefficient of 0.20.
5. **Classification and calibration:** Predict H/L/P probabilities, average predictions from five independently trained models, and apply temperature scaling fitted on validation data.

The final model does not use supervised contrastive learning (SupCon).

## Dataset and evaluation

The study uses the **KidLearn-Engage Corpus**, comprising **401 recordings from 59 students aged 7–14 years**.

| Split | Recordings | Participants | H | L | P |
|---|---:|---:|---:|---:|---:|
| Training | 281 | 41 | 93 | 94 | 94 |
| Validation | 60 | 9 | 20 | 20 | 20 |
| Test | 60 | 9 | 20 | 20 | 20 |

Participants do not overlap across the three splits. Feature standardization, PCA, and quality-feature normalization are fitted using training data only. Temperature scaling uses validation predictions only.

Participant-grouped five-fold cross-validation provides a separate evaluation on the 281-recording training partition, with preprocessing refitted within each training fold.

**Dataset access:** [Add the confirmed dataset access procedure or approved download link before publishing this README.]

## Reported results

The following values are reported in the accompanying manuscript.

| Evaluation protocol | Accuracy (%) | Macro-F1 (%) | Macro AUC | Log Loss |
|---|---:|---:|---:|---:|
| Five individual seeds, mean ± SD | 79.67 ± 4.77 | 79.80 ± 4.64 | 0.9094 ± 0.0237 | 0.6154 ± 0.0704 |
| Five-seed calibrated test ensemble | **83.33** | **83.41** | **0.9346** | **0.4894** |
| Participant-grouped five-fold CV | 73.21 ± 10.77 | 73.32 ± 10.48 | 0.8646 ± 0.0704 | 0.7860 |

The ensemble results refer to the frozen 60-recording test set. Cross-validation uses a separate protocol and shows greater variation across unseen-participant groups. These scores should not be interpreted as performance on an independent external dataset.

## Software

The manuscript reports implementation using:

- Python and PyTorch
- Hugging Face Transformers
- Librosa and SoundFile
- Scikit-learn
- XGBoost for the gradient-boosting baseline
- Google Colab with CUDA acceleration

**Installation:** [Add the tested Python version, dependency versions, and installation commands matching the uploaded code.]

## Reproducing the experiments

The experimental workflow consists of preparing participant-disjoint splits, extracting features, fitting preprocessing on training data, training the models, calibrating the ensemble on validation predictions, and evaluating the frozen test set.

The five main training seeds are **42, 52, 62, 72, and 82**. Main-model training uses AdamW, a batch size of 4 with two-step gradient accumulation, up to 35 epochs, and early-stopping patience of 8. The ablation and grouped-CV experiments use separate training schedules described in the manuscript.

**Execution:** [Add the actual notebook or script names, required input paths, execution order, and training/evaluation commands after uploading the code.]

## Scope and limitations

Evaluation is limited to the study cohort. Performance varies across participant groups, and external-cohort validation is not reported. Predictions are intended for research on speech-based engagement classification and should not be treated as standalone educational assessments.

## Citation

If this code supports your research, please cite the accompanying manuscript:

Sneha Sureddy and I. Jeena Jacob. *WavLM-RAG: Residual Adaptive Gating of Temporal and Global Acoustic Representations for Speech-Based Student Engagement Classification.*

Publication details and DOI can be added when available.

## License

[Add the selected code license and a link to the LICENSE file. If MIT is selected, use: "The original code in this repository is licensed under the MIT License. See LICENSE for details."]

Third-party code, pretrained models, and datasets remain subject to their respective licenses and access conditions.

## Contact

**Sneha Sureddy**, corresponding author  
Email: [sneha.sureddy@gmail.com](mailto:sneha.sureddy@gmail.com)
