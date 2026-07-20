# NetMicrobiome

NetMicrobiome is a Python framework for differential microbiome network analysis based on microbial gene-family co-abundance networks. The framework identifies rewiring events between biological conditions by comparing condition-specific co-abundance networks and functionally characterizes the rewired gene families through pathway enrichment analysis.

The methodology has been developed for the manuscript:

> A Network-Based Framework for Differential Microbiome Analysis Through Gene-Family Co-Abundance Network Rewiring

---

# Overview

NetMicrobiome performs five main steps:

1. Data preprocessing
2. Construction of condition-specific gene-family co-abundance networks
3. Differential network inference
4. Identification of rewired co-abundance associations
5. Functional pathway enrichment analysis

The framework is designed to investigate microbiome rewiring across biological conditions while considering biological stratification factors such as sex.

---

# Workflow

The complete workflow is illustrated below.

(Insert Figure 1 here)

---

# Repository structure

```
NetMicrobiome
│
├── data/
├── scripts/
├── results/
├── figures/
├── docs/
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/mmilano87/NetMicrobiome
cd NetMicrobiome
```

Install the required Python packages

```bash
pip install -r requirements.txt
```

---

# Input

NetMicrobiome requires:

- microbial gene-family abundance matrix
- sample metadata
- disease labels
- sex labels

Rows correspond to microbial gene families and columns correspond to samples.

---

# Pipeline

The analysis consists of the following steps.

## Step 1

Construct condition-specific gene-family co-abundance networks using Spearman correlation.

## Step 2

Infer differential networks by comparing male and female co-abundance networks within each condition.

## Step 3

Identify rewired co-abundance associations.

## Step 4

Perform pathway enrichment analysis using the rewired gene families.

## Step 5

Generate summary tables and publication-quality figures.

---

# Output

The framework generates:

- condition-specific co-abundance networks
- differential networks
- rewired gene-family lists
- pathway enrichment tables
- summary statistics
- publication-ready figures

---

# Example

An example dataset together with example outputs is provided in the repository to reproduce the complete workflow.

---

# Reproducibility

The repository contains all scripts required to reproduce the analyses presented in the manuscript.

---

# Citation

If you use NetMicrobiome in your research, please cite:

Milano M. et al.
*A Network-Based Framework for Differential Microbiome Analysis Through Gene-Family Co-Abundance Network Rewiring.*
Briefings in Bioinformatics (under review).

---

# License

This project is released under the MIT License.
