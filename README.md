# MedKGraphFusion

**MedKGraphFusion** is an ontology-guided framework for constructing and expanding biomedical knowledge graphs (KGs) from heterogeneous text using large language models (LLMs). It integrates ontology-guided seed discovery, schema-constrained zero-shot extraction, and concept-centric global fusion with structured verification, enabling scalable biomedical KG construction, prognostic modeling, and patient-level mechanistic interpretation.  

This repository contains code and data to reproduce experiments in our paper: *MedKGraphFusion: Ontology-Guided Knowledge Graph Construction via LLM-Powered Graph Fusion for Heterogeneous Biomedical Texts*.

---

## Features

- **Biomedical KG construction** from heterogeneous literature and structured data  
- **Ontology-guided extraction** ensures domain consistency and improved factual reliability  
- **Schema-constrained triple extraction** using LLMs with zero-shot and local/global fusion strategies  
- **Patient-level interpretability** for linking genes, drugs, and clinical outcomes  

---

## Requirements

- Python 3.10+  
- PyTorch 2.1+  
- Transformers library (Hugging Face)  
- Other dependencies (install via `pip install -r requirements.txt`)  

> **Note:** LLMs used in MedKGraphFusion include Qwen-2.5, Qwen-2.5-72B, Gemma3, LLaMA3.1. Ensure you have access to the models and sufficient GPU memory for larger models (≥70B parameters may require multi-GPU setup or model parallelism).

---
## Usage

Run the main KG construction pipeline with:

```bash
python main.py \
    --run_name "test" \
    --model "llama3.1:70b" \
    --dataset "text_data" \
    --relation_definitions_file "text_data/relation_types_n.json" \
    --gold_concept_file "text_data/database_entities.tsv"
```

> **Arguments:**
--run_name : Name of the current experiment/run
--model : Backbone LLM to use (e.g., "llama3.1:70b", "qwen2.5-14b")
--dataset : Path to input text data directory
--relation_definitions_file : JSON file defining allowed relation types for extraction
--gold_concept_file : TSV file listing gold-standard biomedical entities for evaluation

> **Prompt Template:**
MedKGraphFusion uses a structured LLM prompt for entity and relation extraction:
Outputs strictly in JSON array format of triples: {"s": "<head>", "p": "<relation>", "o": "<tail>"}
Supports both template placeholders and real instance data
Example placeholders used in prompts:
{relation_definitions}
{abstracts}
{concepts}
Full prompt templates are available in the text_data/prompts/ directory.

---

## KG Expansion & Merging

After generating triples using the MedKGraphFusion pipeline, you can dynamically expand an existing knowledge graph (KG) using the scripts in the `KG_merge` folder. This allows integration of newly extracted triples with your current KG.  

Run the merging script as follows:

```bash
python main.py \
    --kg ./KG_merge/MedKG.csv \
    --input ./outputs/step-03.jsonl \
    --output kg_merged.csv \
    --save-extra > output_merger.log
```

> **Arguments:**
--kg : Path to the existing knowledge graph CSV file to be expanded
--input : Path to the JSONL file containing newly extracted triples
--output : Path to save the merged KG
--save-extra : Flag to save additional metadata or debug info
> : Redirects logs to output_merger.log for tracking
This step ensures seamless integration of new information into your existing MedKG, supporting iterative KG construction and continuous updates.
