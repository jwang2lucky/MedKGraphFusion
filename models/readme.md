# Model Download Guide

This repository does **not** include the pretrained model weights due to file size and license considerations.

Before running the code, please manually download the following models from Hugging Face and place them in the corresponding local directories.

## Required Models

The project expects the following folder structure:

```bash
models/
├── BioLORD/
├── sapbert/
├── bio_clinicalbert/
└── all-mpnet-base-v2/

Please download the pretrained models from the following Hugging Face pages:

BioLORD
Hugging Face:
https://huggingface.co/FremyCompany/BioLORD-2023

SapBERT
Hugging Face:
https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext

Bio_ClinicalBERT
Hugging Face:
https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT

all-mpnet-base-v2
Hugging Face:
https://huggingface.co/sentence-transformers/all-mpnet-base-v2