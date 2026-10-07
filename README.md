# Dental-Implant-Classification

This repository contains the official implementation for the dental implant classification model using the AI-Hub public dataset.

## Project Overview
- **Objective:** Open-set classification and specification estimation of dental implants from radiographic images.
- **Dataset:** [AI-Hub Radiographic Image Data of Dental Implants](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=536)
- **Framework:** PyTorch

Deep learning models for dental implant identification are usually trained and evaluated on a fixed set of implant systems. In practice, however, new systems keep entering the market, and a closed-set model silently assigns every unregistered implant to a known system, often with high confidence.

This repository contains the code for our study, which evaluates implant identification under open-set conditions: the model must identify registered systems and flag implants from systems it has never seen.


## Code repository
- common.py	Shared dataset, image transforms, and backbone utilities.
- arpl.py	ARPL loss following the official implementation.
- openmax.py	OpenMax and MAV distance scoring on classifier logits.
- osr_metrics.py	Open-set evaluation metrics, including AUROC, OSCR, and threshold-based scores.
- train_ce.py	Trains the cross-entropy classifier and saves logits, features, and ODIN scores.
- train_arpl.py	Trains the ARPL model and saves its outputs and evaluation metrics.
- eval_baselines.py	Compares all open-set methods on identical splits with paired statistical tests.
- check_openmax.py	Runs OpenMax sanity checks and hyperparameter sensitivity analysis.
- analyze_all.py	Aggregates subgroup and error analyses over all splits into paper-ready tables.
- aggregate.py	Summarizes per-split metrics as mean ± standard deviation.
All experiments were run on Google Colab with a single NVIDIA Tesla T4 GPU. Training takes about one hour per split; all evaluation and figure scripts run on CPU.

### Model Weights
Download the pre-trained weights from the [Release page].
Please place the file in the 'weights/' directory before inference.
