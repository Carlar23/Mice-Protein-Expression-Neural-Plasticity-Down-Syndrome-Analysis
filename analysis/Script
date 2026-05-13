#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Carlos
"""

# =====================================
#            0. DATA LOAD
# =====================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


os.getcwd()
os.chdir("/Users/Carlos/Desktop/")
df = pd.read_csv("mice_protein_expression.csv", sep=";")

# =====================================
#            1. DATA CLEANING
# =====================================


# 1. Preliminary inspection:
# --------------------------

df.head()          # first 5 rows
df.shape           # number of rows and columns
df.columns         # column names
df.info()          # data types and missing values
df.describe()      # basic statistics (mean, std, min, max...)

# 2. Missing values:
# ------------------
df.isnull().sum()                        # nulls per column
df.isnull().sum() / len(df) * 100        # percentage of nulls


# 3. Distribution of numerical variables:
# ----------------------------------------

df.hist(figsize=(12, 8), bins=30)
sns.histplot(df['BDNF_N'], kde=True)     # for a specific variable


# 4. Distribution of categorical variables:
# ------------------------------------------

df['class'].value_counts()                   # frequency of each value
df['class'].value_counts(normalize=True)     # as percentage
sns.countplot(x='class', data=df)            # visualize distribution of categorical variables


# 5. Correlations:
# ----------------

df.corr(numeric_only=True)               # correlation matrix using only numerical variables
# visual heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')  # correlation heatmap for numerical variables


# 6. Relationship between variables:
# ------------------------------------

# two numerical variables
sns.scatterplot(x='DYRK1A_N', y='ITSN1_N', data=df)


# 7. Outlier detection:
# ---------------------
# using IQR
Q1 = df['BDNF_N'].quantile(0.25)
Q3 = df['BDNF_N'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['BDNF_N'] < Q1 - 1.5 * IQR) |
              (df['BDNF_N'] > Q3 + 1.5 * IQR)]



# =========================================
#         2. UNSUPERVISED METHODS
# =========================================

# 1. Principal Component Analysis (PCA):
# ---------------------------------------
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# 1. Select only numerical columns
df_num = df.select_dtypes(include='number')

# 2. Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_num)


# 3. Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)


# 4. Explained variance per component:
varianza = pca.explained_variance_ratio_
print(varianza)

# The first two dimensions ([0.39744098, 0.30236753]) capture 70% of the dataset information.

print(np.cumsum(varianza))   # cumulative variance

# 5. Scree plot for choosing the number of components:
plt.figure(figsize=(8, 4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.axhline(y=0.95, color='r', linestyle='--', label='95%')
plt.legend()


# 6. PCA with a fixed number of components
pca2 = PCA(n_components=2)
X_pca2 = pca2.fit_transform(X_scaled)

# convert to dataframe
df_pca = pd.DataFrame(X_pca2, columns=['PC1', 'PC2'])
print(df_pca.head())


# 7. 2D visualization
df_pca['class'] = df['class'].values

plt.figure(figsize=(8, 6))
for group in df_pca['class'].unique():
    subset = df_pca[df_pca['class'] == group]
    plt.scatter(subset['PC1'], subset['PC2'], label=group, alpha=0.6)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.title('PCA - 2D Projection')


# 8. Variable contribution to each component:
loadings = pd.DataFrame(
    pca2.components_.T,
    columns=['PC1', 'PC2'],
    index=df_num.columns
)
print(loadings.sort_values('PC1', ascending=False))


# Interpretation:
'''
With only 2 components, 70% of the dataset information is captured. This indicates
that the 14 proteins have a clear underlying structure. Furthermore, to obtain
95% of the variance, 7 out of 14 components would be needed, suggesting
high correlation among the proteins.
'''


# 2. t-SNE:
# ---------
from sklearn.manifold import TSNE

# 1. Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

# 2. Convert to dataframe
df_tsne = pd.DataFrame(X_tsne, columns=['TSNE1', 'TSNE2'])


# 3. Plot without labels
plt.figure(figsize=(8, 6))
plt.scatter(df_tsne['TSNE1'], df_tsne['TSNE2'], alpha=0.5)
plt.title('t-SNE')
plt.xlabel('TSNE1')
plt.ylabel('TSNE2')


# 4. Plot with categorical variable
df_tsne['class'] = df['class'].reset_index(drop=True)

plt.figure(figsize=(8, 6))
for group in df_tsne['class'].unique():
    subset = df_tsne[df_tsne['class'] == group]
    plt.scatter(subset['TSNE1'], subset['TSNE2'], label=group, alpha=0.6)

plt.legend()
plt.title('t-SNE by groups')
plt.savefig('tsne_groups.png')




# =========================================
#              3. CLUSTERING
# =========================================

# 1. K-Means:
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

df['cluster'] = kmeans.labels_
print(df['cluster'].value_counts())


# Choosing the number of clusters (Elbow Method)
inertias = []
k_range = range(1, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k_range, inertias, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.savefig('elbow_method.png')


# Mean of each variable per cluster
profile = df.groupby('cluster').mean(numeric_only=True)
print(profile)

# Cluster profile heatmap
plt.figure(figsize=(12, 4))
sns.heatmap(profile, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Cluster profile')






# =========================================
#          4. SUPERVISED METHODS
# =========================================

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 1. Support Vector Machine (SVM)
# --------------------------------

# 1. Select variables
X = df_num.drop('BDNF_N', axis=1)   # predictor variables
Y = df['class']                      # target variable (categorical)


# 2. Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# 3. Scale the training data
scales = StandardScaler()
X_train = scales.fit_transform(X_train)
X_test = scales.transform(X_test)


# 4. Train the model
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X_train, Y_train)


# 5. Model predictions
Y_pred = svm.predict(X_test)
print(classification_report(Y_test, Y_pred))


# 6. Confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=svm.classes_,
            yticklabels=svm.classes_)
plt.xticks(rotation=45)
plt.title('Confusion Matrix — SVM')
plt.tight_layout()


# Interpretation
'''
Overall, the classifier distinguishes the different classes very well, since model
precision exceeds 85% for all classes except Control-Non-learning. This allows us
to infer the role of protein expression as a differentiating element across classes.
'''


# 2. Random Forest
# ----------------

from sklearn.ensemble import RandomForestClassifier

# 1. Select variables
X = df_num.drop('BDNF_N', axis=1)    # predictor variables
Y = df['class'][X.index]             # target variable (categorical)


# 2. Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# 3. Train the model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, Y_train)


# 4. Model predictions
Y_pred = rf.predict(X_test)
print(classification_report(Y_test, Y_pred))


# 5. Confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=rf.classes_,
            yticklabels=rf.classes_)
plt.xticks(rotation=45)
plt.title('Confusion Matrix — Random Forest')
plt.tight_layout()


# 6. Feature importance
importances = pd.Series(
    rf.feature_importances_,
    index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x=importances.values, y=importances.index)
plt.title('Protein importance')
plt.tight_layout()



# =========================================
#             5. ROC CURVES
# =========================================

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# 1. Binarize classes (One-vs-Rest)
classes = sorted(Y.unique())
Y_test_bin = label_binarize(Y_test, classes=classes)

# 2. Retrain SVM with probability=True
# (X_train and X_test are already scaled from the previous step)
svm_prob = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)
svm_prob.fit(X_train, Y_train)
Y_prob = svm_prob.predict_proba(X_test)

# 3. Compute and plot ROC curve for each class
plt.figure(figsize=(8, 6))

for i, cls in enumerate(classes):
    fpr, tpr, _ = roc_curve(Y_test_bin[:, i], Y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{cls} (AUC = {roc_auc:.2f})')

# 4. Reference line (random classifier)
plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)')

plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve — SVM (One-vs-Rest)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_svm.png')

'''
The ROC curves confirm and reinforce the classification results. The SVM model
achieves an AUC above 0.95 for all classes, indicating excellent discriminatory
capacity regardless of the decision threshold used. The class with the lowest AUC
is predictably Trisomic-Non-learning, in line with the F1 = 0.84 obtained previously,
which reinforces the hypothesis of molecular heterogeneity within that group.
'''


# =========================================
#          GENERAL INTERPRETATION
# =========================================

'''
The results demonstrate that the cortical protein expression profile allows
high-precision discrimination of mouse genotype and learning ability.
Random Forest achieved 92% accuracy and SVM 89%, confirming that molecular
information alone is sufficient to classify the four clinical groups.
Not all proteins contribute equally: pERK_N (18.9%), ITSN1_N (13.3%) and
DYRK1A_N (13.1%) concentrate the highest discriminatory power.
The hardest group to classify was Trisomic-Non-learning (F1 = 0.84), which
may reflect molecular heterogeneity within that phenotype. Overall, the results
are consistent with the literature, as the most relevant proteins are involved
in the MAPK pathway, synaptic plasticity, and trisomy of chromosome 21.
'''
