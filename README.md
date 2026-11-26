# Exploring the Multidimensional Structure of Well-Being Through Network Analysis

## Overview

- This repository contains the code and supporting network data accompanying the research paper "Exploring the Multidimensional Structure of Well-Being Through Network Analysis."

- **Goal:** The study aims to develop a comprehensive understanding of how different attributes and domains are interconnected and contribute to the overall structure of well-being using a network-based approach. Drawing on the first wave of the Global Flourishing Study, the study consideres a broad conceptualization of well-being encompassing happiness and life satisfaction, physical and mental health, meaning and purpose, character and virtue, close social relationships, and financial and material stability alongside related aspects of human experience.
  
- **Network Construction:** Two well-being networks are constructed using regularized Gaussian Graphical Models (GGMs):

  - Overall well-being network: Estimated from correlations computed across the full sample.

  - Pooled within-country network: Estimated by pooling correlations computed separately within each country to account for between-country differences.

  In both networks, nodes represent individual attributes, and weighted edges capture conditional associations between them.

- **Network Analysis:** The overall network is examined using connectivity analysis, node and bridge centrality measures, and community detection to identify core attributes, cross-domain connections, and the empirical organization of well-being. The connectivity and centrality patterns of the two networks are then compared to assess the robustness of the findings to between-country differences.


- **Original Survey Dataset:** The study uses Wave 1 of the Global Flourishing Study (GFS), which contains survey responses from 202,897 participants across 22 countries. A subset of 48 attributes spanning seven conceptual domains was selected for network construction and analysis.

  - The dataset is publicly available through the Center for Open Science: [Global Flourishing Study](https://www.cos.io/gfs-wave-data)
  - The complete list of selected attributes, their descriptions, response scales, and preprocessing procedures are provided in the accompanying manuscript.

- **Derived Network Data:** This repository provides the network data generated from the selected survey attributes, including the overall well-being network and the pooled within-country network.

  - Both networks are weighted and undirected, with 48 nodes representing individual survey attributes and edges representing regularized partial correlations.
  - The network data are provided in CSV format and can be used to reproduce the network analyses.
 
- **Code Overview:** The repository contains four Python scripts implementing overall network construction, pooled within-country network construction, centrality analysis, and community detection.


---


## Dataset


- **Original Survey Data:** The data can be obtained from the Center for Open Science: [Global Flourishing Study](https://www.cos.io/gfs-wave-data).
- **Derived Network Data:** The `data/` directory contains two folders corresponding to the two estimated networks:
    - **Overall well-being network:** The network estimated using survey responses across the full sample.
    - **Pooled within-country network:** The network estimated by pooling correlations computed separately within each country.

    Each folder contains two CSV files:
    - `nodes.csv`: Contains the 48 nodes representing the selected survey attributes and their associated information.
    - `edges.csv`: Contains the connections between pairs of nodes and their corresponding edge weights, representing regularized partial correlations.


---


## Code

The repository contains four Python scripts covering network construction and analysis.

- **Network Construction:**

  - overall-network-construction.py: Constructs the overall well-being network from the original survey data using pairwise Spearman correlations and graphical LASSO regularization. The resulting network is exported as node and edge CSV files.
  - pooled-country-network.py: Constructs the pooled within-country network by calculating Spearman correlations separately within each country, combining them using sample-size-weighted Fisher's z transformation, and applying graphical LASSO regularization. The resulting network is exported as node and edge CSV files.

- **Network Analysis:**

  - centrality.py: Calculates node strength, expected influence, well-being strength, well-being expected influence, bridge strength, bridge expected influence, bridge closeness, and bridge betweenness to identify central attributes and attributes connecting different conceptual domains.
  - community.py: Applies Louvain, Leiden, and Spinglass community detection algorithms to identify groups of interconnected attributes, examine the stability of the detected communities, and assess their alignment with the predefined conceptual domains.

---


## Methodology

The analysis consists of the following main components:

1. **Data preprocessing:** Selection and recoding of the 48 survey attributes, treatment of missing responses, and preparation of the data for network estimation.

2. **Network construction:** Calculation of pairwise Spearman correlations followed by graphical LASSO regularization. The regularization parameter is selected using the Extended Bayesian Information Criterion (EBIC), with gamma = 0.5.

3. **Centrality analysis:** Calculation of node strength, expected influence, well-being-specific strength and expected influence, and bridge centrality measures.

4. **Community detection:** Application of Louvain, Leiden, and Spinglass algorithms to identify communities of interconnected attributes. 

5. **Pooled country network:** Estimation of within-country Spearman correlations followed by sample-size-weighted Fisher's z pooling and regularized network estimation.


---



## Usage Instructions

Before running the scripts, download Wave 1 of the Global Flourishing Study dataset from the official data source (https://www.cos.io/gfs-wave-data). Run the following commands from the main directory using Python 3.

1. Overall Network Construction:
```bash
python3 overall-network-construction.py --input wellbeing-dataset.csv --output data
```

2. Pooled Within-Country Network Construction
```bash
python3 pooled-country-network.py --input wellbeing-dataset.csv --output data/pooled
```

3. Centrality Analysis
```bash
python3 centrality.py
```

4. Community Detection
```bash
python3 community.py
```

The scripts generate CSV files containing the estimated networks and analysis results.


---


## Requirements

- Python 3. 
- Required libraries:
   - NumPy
   - Pandas
   - SciPy
   - Scikit-learn
   - NetworkX
   - Matplotlib.


---


## Research Citation

If you use the code or derived network data provided in this repository, please cite the accompanying research paper:

Alrasheed, H. Exploring the Multidimensional Structure of Well-Being Through Network Analysis. Manuscript submitted for publication.
