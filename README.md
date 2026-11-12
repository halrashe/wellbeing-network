# Exploring the Multidimensional Structure of Well-Being Through Network Analysis

This repository contains the code and supporting network data accompanying the research paper "Exploring the Multidimensional Structure of Well-Being Through Network Analysis."

The study investigates the interconnected structure of well-being using Wave 1 of the Global Flourishing Study (GFS), comprising 202,897 participants from 22 countries. It examines 48 survey attributes organized into seven conceptual domains: Well-being, Health, Religion, Happiness and Life Satisfaction, Character and Virtue, Close Social Relationships, and Childhood.

The analysis uses regularized Gaussian Graphical Models (GGMs) to estimate conditional associations among attributes and applies network analysis techniques to examine centrality, cross-domain connectivity, and community structure.

## Dataset

The study uses publicly available data from Wave 1 of the Global Flourishing Study (https://www.cos.io/gfs-wave-data).

The original survey data can be obtained from the source above. The manuscript provides the complete list of selected attributes, their descriptions, and response scales.

## Code and Methodology

The analysis consists of the following main components:

1. **Data preprocessing:** Selection and recoding of the 48 survey attributes, treatment of missing responses, and preparation of the data for network estimation.

2. **Network construction:** Calculation of pairwise Spearman correlations followed by graphical LASSO regularization. The regularization parameter is selected using the Extended Bayesian Information Criterion (EBIC), with gamma = 0.5.

3. **Centrality analysis:** Calculation of node strength, expected influence, well-being-specific strength and expected influence, and bridge centrality measures.

4. **Community detection:** Application of Louvain, Leiden, and Spinglass algorithms to identify communities of interconnected attributes. 

5. **Pooled within-country network:** Estimation of within-country Spearman correlations followed by sample-size-weighted Fisher's z pooling and regularized network estimation.

6. **Robustness analysis:** Evaluation of network estimates and centrality stability using bootstrap resampling and comparison of pairwise-deletion and complete-case approaches.

## Usage Instructions



## Requirements

The code requires Python 3 and the following Python libraries:NumPy, Pandas, SciPy, Scikit-learn (for graphical LASSO network estimation), NetworkX (for network analysis), and Matplotlib.

The implementation uses graphical_lasso from sklearn.covariance for regularized network estimation and ConvergenceWarning from sklearn.exceptions for handling convergence warnings.

