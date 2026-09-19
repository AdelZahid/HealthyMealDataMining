# Machine Learning-Based Healthy Meal Classification and Nutritional Pattern Analysis

A Java and WEKA-based data mining system for **healthy meal classification and nutritional pattern analysis**. The project integrates preprocessing, exploratory data analysis, supervised classification, class-imbalance handling, clustering, association-rule mining, pattern analysis, and model persistence into a complete data mining workflow.

---

## 📌 Overview

This project applies machine learning and data mining techniques to analyze nutritional and categorical meal information and classify meals according to their healthiness.

The system is designed as a modular data mining pipeline that can:

* Load and inspect CSV datasets
* Preprocess missing and categorical values
* Analyze dataset statistics and distributions
* Handle class imbalance using SMOTE
* Train and evaluate multiple classification models
* Compare model performance using multiple evaluation metrics
* Analyze feature importance
* Discover nutritional clusters using K-Means
* Discover meaningful relationships using Apriori association rules
* Identify important nutritional patterns
* Save and reload trained models
* Predict the class of new meal records
* Generate detailed analysis reports

The current implementation uses a **Healthy Meal Plans dataset containing 500 instances and 14 attributes**.

---

## ✨ Key Features

### 📂 Dataset Inspection

The system automatically analyzes:

* Number of instances
* Number of attributes
* Attribute types
* Missing values
* Duplicate records
* Distinct values
* Numeric ranges
* Class attribute
* Class distribution
* Class imbalance ratio

Example dataset:

```text
Instances       : 500
Attributes      : 14
Missing Values  : 0
Duplicate Rows  : 0
Class Attribute : is_healthy
```

---

### 🧹 Data Preprocessing

The preprocessing pipeline supports:

* Missing-value handling
* String-to-nominal conversion
* Numeric normalization
* Attribute preparation
* Class attribute configuration
* SMOTE-based class balancing when required

The preprocessing pipeline is designed to avoid unnecessary data leakage during model evaluation.

---

### 📊 Exploratory Data Analysis

The project analyzes:

* Dataset statistics
* Class distribution
* Class imbalance
* Numeric feature ranges
* Feature relationships
* Correlations
* Outliers
* Feature importance

These analyses provide a better understanding of the dataset before model training.

---

## 🤖 Machine Learning Models

The classification module evaluates multiple supervised learning algorithms:

### Random Forest

An ensemble of decision trees used for robust classification and feature analysis.

### J48 Decision Tree

A C4.5-style decision tree classifier that provides interpretable classification rules.

### SMO

Support Vector Machine-based classification implemented through WEKA's Sequential Minimal Optimization.

---

## ⚖️ Class Imbalance Handling

The dataset contains a significant class imbalance.

Example:

```text
Class 0 : 453 samples (90.60%)
Class 1 :  47 samples (9.40%)

Imbalance Ratio : 9.64 : 1
```

To address substantial imbalance, the system supports **SMOTE (Synthetic Minority Oversampling Technique)**.

SMOTE is applied within the training process to reduce the risk of data leakage during model evaluation.

---

## 📈 Model Evaluation

The classification models are evaluated using **stratified 10-fold cross-validation**.

The system reports:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix

Example result from the current dataset:

```text
Best Model : J48

Accuracy : 97.40%
F1-Score : 85.06%
ROC-AUC  : 91.22%
```

> The displayed values are examples from the current dataset and may change when another dataset is analyzed.

---

## 🔍 Feature Importance

The system provides feature-importance analysis to identify the variables that contribute most strongly to classification.

Depending on the analysis, feature importance can be derived from:

* Permutation importance
* Decision-tree split attributes
* Model-specific importance measures

This helps interpret which nutritional or categorical features are most relevant to meal classification.

---

## 🧩 K-Means Clustering

The project uses **K-Means clustering** to discover groups of meals based on nutritional characteristics.

The clustering workflow includes:

1. Preprocessing nutritional attributes
2. Evaluating candidate values of `K`
3. Calculating within-cluster SSE
4. Using an elbow-based approach to select a defensible `K`
5. Generating cluster assignments
6. Profiling the nutritional characteristics of each cluster
7. Comparing clusters with the healthiness class

Example workflow:

```text
K = 1
K = 2
K = 3
K = 4
K = 5
K = 6
      ↓
Elbow Analysis
      ↓
Selected K
      ↓
Cluster Profiles
```

---

## 🔗 Apriori Association Rule Mining

The system uses **Apriori** to discover relationships among nutritional attributes.

Selected numeric attributes can be discretized into categories such as:

```text
Low
Medium
High
```

The generated association rules are evaluated using:

* Support
* Confidence
* Lift

The analysis emphasizes meaningful rules related to the detected class attribute.

Example:

```text
Protein = High
+
Calories = Low
        ↓
is_healthy = 1
```

High-lift rules are highlighted because they indicate stronger-than-baseline associations.

---

## 🧠 Pattern Analysis

The system integrates findings from:

* Classification
* Feature importance
* Clustering
* Association rules
* Class distribution

to identify important nutritional and healthiness-related patterns.

This provides a higher-level interpretation of the results instead of relying only on individual machine-learning metrics.

---

## 💾 Model Persistence

The project supports trained-model persistence.

The workflow is:

```text
Train Model
     ↓
Select Best Model
     ↓
Serialize Model
     ↓
Save Model
     ↓
Reload Model
     ↓
Predict New Data
```

This allows the trained model to be reused without retraining every time.

---

## 🔮 New Meal Prediction

After training, the system can load the persisted model and predict the class of a new meal record.

Example workflow:

```text
New Meal Data
     ↓
Preprocessing
     ↓
Loaded Model
     ↓
Prediction
     ↓
Predicted Class
```

---

# 🏗️ Project Architecture

```text
HealthyMealDataMining/
│
├── src/
│   └── main/
│       └── java/
│           └── dwm/
│               ├── Main.java
│               ├── Util.java
│               ├── DataLoader.java
│               ├── DataInspector.java
│               ├── DataPreprocessor.java
│               ├── EDAAnalyzer.java
│               ├── ClassificationEngine.java
│               ├── ClusteringEngine.java
│               ├── AssociationRuleMiner.java
│               ├── PatternAnalyzer.java
│               ├── ModelPersistence.java
│               └── MealPredictor.java
│
├── data/
│   └── healthy_meal_plans.csv
│
├── lib/
│   ├── weka.jar
│   └── SMOTE.jar
│
├── models/
│
├── results/
│
├── target/
│
├── pom.xml
└── README.md
```

---

# 🔄 Complete Data Mining Workflow

```text
                    CSV Dataset
                         │
                         ▼
                ┌─────────────────┐
                │ Data Inspection │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Preprocessing   │
                │ Missing Values  │
                │ Categorical     │
                │ Normalization   │
                └────────┬────────┘
                         │
              ┌──────────┼───────────┐
              │          │           │
              ▼          ▼           ▼
            EDA    Classification  K-Means
              │          │           │
              │          ▼           │
              │       SMOTE          │
              │          │           │
              │          ▼           │
              │    Model Evaluation  │
              │          │           │
              │          ▼           │
              │    Best Model        │
              │          │           │
              └──────────┼───────────┘
                         │
                         ▼
                Apriori Association
                     Rules
                         │
                         ▼
                  Pattern Analysis
                         │
                         ▼
                  Model Persistence
                         │
                         ▼
                   New Prediction
                         │
                         ▼
                   Final Reports
```

---

# 🛠️ Technologies Used

| Technology    | Purpose                           |
| ------------- | --------------------------------- |
| Java 17       | Core application development      |
| WEKA 3.8.6    | Machine learning and data mining  |
| SMOTE 1.0.3   | Class imbalance handling          |
| Maven         | Project and dependency management |
| IntelliJ IDEA | Development environment           |
| CSV           | Dataset format                    |
| Git/GitHub    | Version control                   |

---

# 📦 Requirements

Before running the project, install:

* **Java JDK 17**
* **IntelliJ IDEA** or another Java IDE
* **Maven** / IntelliJ Maven integration
* **WEKA 3.8.6**
* **SMOTE package**
* Git (optional, for version control)

Verify Java:

```bash
java -version
```

Expected:

```text
java version "17.x.x"
```

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/HealthyMealDataMining.git
```

Navigate into the project:

```bash
cd HealthyMealDataMining
```

---

## 2. Open the project

Open the project in IntelliJ IDEA.

Make sure the project SDK is configured to:

```text
Java 17
```

---

## 3. Verify the libraries

The project uses:

```text
lib/weka.jar
lib/SMOTE.jar
```

Make sure both files are available before running the application.

---

# ▶️ Running the Application

Run:

```text
dwm.Main
```

from IntelliJ IDEA.

The application will:

```text
Load Dataset
      ↓
Inspect Dataset
      ↓
Preprocess Data
      ↓
Perform EDA
      ↓
Train Classifiers
      ↓
Evaluate Models
      ↓
Perform Clustering
      ↓
Mine Association Rules
      ↓
Analyze Patterns
      ↓
Save Best Model
      ↓
Generate Reports
```

Generated reports are stored in:

```text
results/
```

Trained models are stored in:

```text
models/
```

---

# 📁 Dataset

The current dataset is:

```text
data/healthy_meal_plans.csv
```

The dataset contains nutritional and categorical meal information.

Example attributes include:

```text
meal_name
num_ingredients
calories
prep_time
protein
fat
carbs
vegan
vegetarian
keto
paleo
gluten_free
mediterranean
is_healthy
```

The target/class attribute in the current dataset is:

```text
is_healthy
```

---

# 📊 Current Dataset Summary

| Property          |    Value |
| ----------------- | -------: |
| Instances         |      500 |
| Attributes        |       14 |
| Missing Values    |        0 |
| Duplicate Records |        0 |
| Majority Class    |      453 |
| Minority Class    |       47 |
| Imbalance Ratio   | 9.64 : 1 |

---

# 📋 Generated Reports

The system generates text-based reports covering:

* Dataset inspection
* Exploratory data analysis
* Classification
* Model evaluation
* Feature importance
* Clustering
* Association rules
* Pattern analysis
* Final integrated analysis

The reports are saved under:

```text
results/
```

---

# 🔬 Reproducibility

The project uses a fixed random seed for reproducible machine-learning experiments.

Where applicable, the same preprocessing and evaluation configuration is maintained across experiments.

The classification evaluation uses:

```text
10-Fold Stratified Cross-Validation
```

---

# 📌 Important Notes

### Dataset Compatibility

The system is primarily designed for structured classification datasets with:

* Numeric attributes
* Nominal/categorical attributes
* A class/target attribute

The class attribute should be identifiable either automatically or through configuration.

### Class Imbalance

SMOTE should be used when substantial class imbalance exists. It should be applied only to training data during model evaluation to avoid information leakage.

### Reports

The current version generates TXT-based analysis reports. These reports can be further integrated into a graphical dashboard for interactive visualization.

---

# 🚧 Future Improvements

Planned improvements include:

* [ ] Responsive web-based dashboard
* [ ] Drag-and-drop dataset upload
* [ ] Interactive charts and visualizations
* [ ] Real-time analysis progress
* [ ] Interactive confusion matrices
* [ ] Correlation heatmaps
* [ ] Interactive clustering visualization
* [ ] Interactive Apriori rule exploration
* [ ] Automatic report parsing
* [ ] HTML/PDF report generation
* [ ] Dataset-independent dynamic UI
* [ ] REST API integration
* [ ] Web-based model prediction
* [ ] Dark/light mode
* [ ] Mobile-responsive interface

---

# 🎯 Project Objective

The primary objective of this project is to demonstrate how **machine learning and data mining techniques can be integrated into a complete analytical workflow** for nutritional meal classification and pattern discovery.

The project combines predictive modeling with unsupervised learning and association-rule mining to provide both:

**Prediction**

and

**Pattern Discovery**

from nutritional data.

---

# 👨‍💻 Authors

### Adel Mohammad Zahid

B.Sc. in Computer Science and Engineering
Ahsanullah University of Science and Technology (AUST)
Dhaka, Bangladesh

Research interests include:

* Machine Learning
* Deep Learning
* Computer Vision
* Generative AI
* Agentic AI
* Algorithm Design
* Software Development

### MD. Rubayet Islam AL Sahab

B.Sc. in Computer Science and Engineering
Ahsanullah University of Science and Technology (AUST)
Dhaka, Bangladesh

---

# 🎓 Academic Project

**Course:** Data Mining and Warehousing Lab

**Institution:**
Ahsanullah University of Science and Technology (AUST)

---

# 📄 License

This project was developed for academic and educational purposes.

If you reuse or extend this project, please provide appropriate attribution to the original authors.

---

# ⭐ Acknowledgements

* WEKA Machine Learning Software
* SMOTE package for WEKA
* Ahsanullah University of Science and Technology
* Department of Computer Science and Engineering
