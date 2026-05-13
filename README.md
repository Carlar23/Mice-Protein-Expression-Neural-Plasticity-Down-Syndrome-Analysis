# Mice-Protein-Expression-Neural-Plasticity-Down-Syndrome-Analysis
Full machine learning pipeline on cortical protein expression data to classify mouse genotype (trisomic vs. control) and learning ability using both supervised and unsupervised methods.

**Overview**
This project analyzes the Mice Protein Expression dataset through a complete data science pipeline: from data cleaning and exploratory analysis to multiclass classification with high-performance models. The goal is to determine whether the expression profile of 14 cortical proteins is sufficient to distinguish four clinical groups.

|---------------------------------------|
| Group                               N |

| Control-Learning                   159|
| Control-Non-learning               159|
| Trisomic-Learning                  133|
| Trisomic-Non-learning              149|


**Objectives**

1. Explore data structure through Principal Component Analysis (PCA) and t-SNE
2. Detect natural groupings in the data using K-Means clustering
3. Classify the four clinical groups with SVM and Random Forest
4. Identify proteins with the highest discriminatory power
5. Evaluate model performance using standard metrics and ROC curves

 **Methods**
1. Data Cleaning & EDA

Inspection of missing values, duplicates, and distributions
Outlier detection using IQR
Protein correlation heatmap

2. Unsupervised Methods

PCA: dimensionality reduction to explore the underlying data structure
t-SNE: non-linear 2D visualization of clinical groups

3. Clustering

K-Means with optimal cluster selection via the elbow method

4. Supervised Methods

SVM (RBF kernel)
Random Forest (100 estimators)
Evaluation with classification_report, confusion matrices, and ROC curves



**Key Results**
|---------------------------------------|
| Model                        Accuracy |
|---------------------------------------|
| SVM (RBF kernel)                89%   |
| Random Forest                   92%   |
|---------------------------------------|

Most Discriminant Proteins (Random Forest)
---------------------------------------------------------------------------
Protein             Importance             Biological Relevance

pERK_N               18.9%                    MAPK pathway — 
                                              synaptic plasticity 
                                              and memory
ITSN1_N              13.3%                    Chromosome 21: trisomy marker

DYRK1A_N             13.1%                    Chromosome 21: trisomy marker

pCAMKII_N            8.9%                     Memory consolidation

PKCA_N               6.9%                     Neuronal signaling
----------------------------------------------------------------------------

**Key Findings
**1. With only 2 principal components, 70% of the variance is retained, indicating strong collinearity among proteins and a clear underlying biological structure.
2. K-Means spontaneously recovers a structure close to the clinical groups without using labels, validating that the separation is biologically real.
3. The hardest class to classify is Trisomic-Non-learning (F1 = 0.84), suggesting molecular heterogeneity within that phenotype.
4. Results are consistent with the literature: the most relevant proteins are involved in the MAPK pathway, synaptic plasticity, and trisomy of chromosome 21.


**Requirements**
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter



**Dataset**
Mice Protein Expression Data Set

Source: UCI Machine Learning Repository
600 samples · 14 proteins · 4 classes · no missing values


**Author**
Carlos
Master's in Bioinformatics — UNIR
