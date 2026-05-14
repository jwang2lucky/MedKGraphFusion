MedKGraphFusion
MedKGraphFusion is an ontology-guided framework for constructing and expanding biomedical knowledge graphs (KGs) from heterogeneous text using large language models (LLMs). It integrates ontology-guided seed discovery, schema-constrained zero-shot extraction, and concept-centric global fusion with structured verification, enabling scalable biomedical KG construction, prognostic modeling, and patient-level mechanistic interpretation.

This repository contains code and data to reproduce experiments in our paper: MedKGraphFusion: Ontology-Guided Knowledge Graph Construction via LLM-Powered Graph Fusion for Heterogeneous Biomedical Texts.

Features
Biomedical KG construction from heterogeneous literature and structured data.
Ontology-guided extraction ensures domain consistency and improved factual reliability.
Schema-constrained triple extraction using LLMs with zero-shot and local/global fusion strategies.
Patient-level interpretability for linking genes, drugs, and clinical outcomes.
Requirements
Python 3.10+
PyTorch 2.1+
Transformers library (Hugging Face)
Other dependencies (can be installed via pip install -r requirements.txt)
Note: LLMs used in MedKGraphFusion include Qwen-2.5, Qwen-2.5-72B, Gemma3, LLaMA3.1. Make sure you have access to the models and sufficient GPU memory for larger models (≥70B parameters may require multi-GPU setup or model parallelism).