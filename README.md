# NetMicrobiome
microbiome-network-analysis
# Microbiome Differential Network Analysis

This repository implements a pipeline for microbiome network analysis based on gene family interactions.

## Pipeline

1. Network construction from gene family abundance
2. Differential network analysis between conditions
3. Pathway enrichment analysis

## Data

Input data should include:

- Gene family abundance matrix (.mtx)
- Pathway abundance matrix (.mtx)
- Metadata file

## Usage

Run scripts in order:

```bash
python src/build_networks.py
python src/differential_network.py
python src/pathway_enrichment.py

Requirements

Install dependencies:

pip install -r requirements.txt
