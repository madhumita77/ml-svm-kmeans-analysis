# Machine Learning Programming Assignment - SVM & K-Means Clustering

This repository contains implementations of Support Vector Machines (SVM) and K-Means clustering algorithms as part of CS60050 Machine Learning coursework.  The project demonstrates advanced ML techniques, comprehensive model evaluation, and professional software development practices.

## 📋 Project Overview

### Part A: Support Vector Machines (HIGGS Dataset)
- **Dataset**: HIGGS Dataset from UCI Repository (110,000 samples, 28 features)
- **Features**: 28 physics-derived features from particle collision events
- **Task**: Binary classification (Signal vs. Background detection)
- **Techniques**: Linear SVM, Polynomial kernels, RBF kernels, Custom kernels

### Part B: K-Means Clustering (Anuran Calls Dataset)
- **Dataset**: Anuran Calls Dataset (MFCCs)
- **Features**: 22 MFCC coefficients for frog calls
- **Task**: Clustering frog species based on acoustic features
- **Techniques**: K-Means, Hierarchical Clustering, DBSCAN

## 🚀 Quick Start

### Prerequisites
pip install -r Part_A/requirements.txt
pip install -r Part_B/requirements.txt


### Download HIGGS Dataset
Download from UCI repository (11GB)
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
gunzip HIGGS.csv.gz
mv HIGGS.csv data/

### Running Part A (SVM)
**Basic execution with sampling:**
python src/partA_higgs_svm.py --data data/HIGGS.csv --sample-frac 0.01 --k-best 10 --random-state 42

**Fast execution (CI/testing):**
python src/partA_higgs_svm.py --data data/HIGGS.csv --no-eda --no-shap



### Running Part B (K-Means)
**Standard execution:**
python src/partB_KMeans.py --data data/Frogs_MFCCs.csv --random_state 42 --pca_components 10 --k 4 --dbscan_eps 0.5 --dbscan_min_samples 5


## 📁 Project Structure

```bash
├── data/ # Dataset files
│ ├── HIGGS.csv # HIGGS dataset for SVM
│ └── Frogs_MFCCs.csv # Anuran calls dataset for clustering
├── outputs/ # Generated outputs
│ ├── part_A/ # SVM results
│ │ ├── eda/ # Data analytics
│ │ ├── feature selection/ #Top K best features
│ │ ├── hyperparam/ # Hyperparameters
│ │ ├── kernel_comparison/ #Different Kernel Comparisons metrics
│ │ ├── models/ # saved models
│ │ ├── shap/ # SHAP analysis for interpretability
│ │ └── metrics/ # Performance metrics
│ └── part_B/ # K-Means results
│ ├── artifacts/ # Clustering artifacts
│ ├── figures/ # Visualizations
│ ├── metrics/ # Evaluation metrics
│ └── models/ # Saved models
├── Part_A/ # Part A implementation
│ ├── PartA_SVM.ipynb # Jupyter notebook
│ └── requirements.txt # Dependencies for Part A
├── Part_B/ # Part B implementation
│ ├── PartB_KMeans.ipynb # Jupyter notebook
│ └── requirements.txt # Dependencies for Part B
├── reports/ # Analysis reports
│ ├── PartA_Report.pdf # SVM analysis report
│ └── PartB_Report.pdf # K-Means analysis report
└── src/ # Source code
├── partA_higgs_svm.py # SVM implementation script
└── partB_KMeans.py # K-Means implementation script
```

## 🔬 Technical Implementation

### Part A: Support Vector Machine Features
- **Data Preprocessing**: Standardization, feature engineering, feature selection
- **Kernel Methods**: Linear, Polynomial (degrees 2,3,4), RBF, Custom kernels
- **Hyperparameter Tuning**: Grid Search, Random Search, Bayesian Optimization
- **Evaluation**: Cross-validation, multiple metrics (Accuracy, Precision, Recall, F1, AUC)
- **Interpretability**: SHAP analysis for feature importance
- **Scalability**: SGD-based implementation for large datasets
- **Best Performance**: RBF kernel achieved 64.2% accuracy with optimized parameters

### Part B: K-Means Clustering Features
- **Preprocessing**: Correlation analysis, polynomial feature engineering, PCA
- **Clustering**: K-Means with optimal k selection (Elbow method, Silhouette score)
- **Evaluation**: Davies-Bouldin Index, Calinski-Harabasz Index
- **Visualization**: PCA/t-SNE for dimensionality reduction and plotting, elbow method
- **Comparison**: Agglomerative Hierarchical Clustering, DBSCAN
- **Feature Analysis**: MFCC contribution to cluster separation
- **Results**: K-Means optimal at k=4 with 0.401 silhouette score

## 📊 Technical Highlights

### Advanced ML Techniques
- **Feature Selection**: SelectKBest with f_classif scoring
- **Dimensionality Reduction**: PCA with explained variance analysis
- **Model Interpretability**: SHAP values for explainable AI
- **Hyperparameter Optimization**: Grid search and randomized search
- **Cross-validation**: Stratified k-fold validation

### Software Engineering Practices
- **Modular Design**: Object-oriented pipeline classes
- **CLI Interface**: Argparse for production deployment
- **Error Handling**: Robust exception handling and logging
- **Reproducibility**: Fixed random seeds, configuration management
- **Documentation**: Comprehensive docstrings and type hints

## 📈 Performance Metrics

### SVM Results (HIGGS Dataset)
| Kernel | Accuracy | Precision | Recall | F1-Score | Training Time |
|--------|----------|-----------|---------|----------|---------------|
| RBF (γ=auto) | 0.642 | 0.658 | 0.674 | 0.666 | 61.90s |
| Polynomial (d=2) | 0.641 | 0.655 | 0.682 | 0.668 | 65.66s |
| Linear SGD | 0.631 | 0.630 | 0.620 | 0.630 | Fast |

### Clustering Results (Anuran Calls)
| Algorithm | Silhouette Score | Davies-Bouldin | Calinski-Harabasz |
|-----------|------------------|----------------|-------------------|
| K-Means | 0.401 | 1.284 | 2436.420 |
| Hierarchical | 0.366 | 1.499 | 2366.501 |
| DBSCAN | -0.434 | 1.488 | 23.272 |

## 🛠️ Technologies Used

- **Python 3.10+**
- **Machine Learning**: scikit-learn, pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Model Interpretation**: SHAP
- **Hyperparameter Optimization**: scikit-optimize, optuna
- **Parallel Processing**: joblib
- **Development**: Jupyter Notebook, Google Colab


### Clustering Evaluation
- Silhouette Score optimization
- Multiple clustering quality metrics
- Algorithm comparison and trade-off analysis

## 📝 Reports & Analysis

Detailed analysis reports are available in the `reports/` directory:
- **PartA_Report.pdf**: Complete SVM analysis including kernel comparison, hyperparameter sensitivity, and SHAP interpretation
- **PartB_Report.pdf**: Comprehensive clustering analysis with algorithm comparison and evaluation metrics

## 🎯 Key Learning Outcomes

- **Advanced ML Algorithms**: Deep understanding of SVM kernels and clustering techniques
- **Model Evaluation**: Comprehensive metrics and validation strategies
- **Feature Engineering**: Polynomial features, interaction terms, dimensionality reduction
- **Model Interpretability**: SHAP analysis for explainable machine learning
- **Software Development**: Production-ready code with CLI interfaces
- **Research Skills**: Technical report writing and result interpretation

## 🚀 Usage in Production

The scripts are designed for production use with:
- **CLI interfaces** for easy deployment
- **Configuration management** through JSON files
- **Batch processing** capabilities
- **Error handling** and logging
- **Scalable architecture** for large datasets

## 🤝 Contributing

This project was developed as part of academic coursework. For questions or suggestions:
- Open an issue for bug reports
- Fork and submit pull requests for improvements
- Follow coding standards and include tests

## 📄 License

This project is part of academic coursework for CS60050 Machine Learning. Please respect academic integrity guidelines when referencing this work.

## 📧 Contact

Madhumita Gayen - madhumitagayen07@gmail.com

Project Link: [https://github.com/madhumita77/ml-svm-kmeans-analysis](https://github.com/madhumita77/ml-svm-kmeans-analysis)

---
⭐ **If you found this project helpful, please consider giving it a star!**
