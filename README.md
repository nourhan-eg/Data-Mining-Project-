# Data Mining Project

A data mining project covering three core tasks: **Classification**, **Regression**, and **Association Rule Mining**. Each task uses a real-world dataset and follows a full pipeline: preprocessing, modeling, prediction, evaluation, and visualization.

## Project Tasks

| Task | Problem | Dataset |
|------|---------|---------|
| Classification | Network Intrusion Detection | NSL-KDD |
| Regression | Student Performance Prediction | UCI Student Performance |
| Association Rules | Product Recommendation System | Online Retail (UCI) |

---

## 1. Classification: Network Intrusion Detection

**Goal:** Classify network traffic as normal or as a specific type of cyber-attack.

**Dataset:** NSL-KDD

**Evaluation metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC

## 2. Regression: Student Performance Prediction

**Goal:** Predict a student's final grade (G3) from study habits and demographic factors.

**Dataset:** UCI Student Performance

**Evaluation metrics:** MSE / RMSE, R² Score

## 3. Association Rules: Product Recommendation System

**Goal:** Discover products that are frequently bought together and use the rules to recommend items based on customer purchasing patterns.

**Dataset:** Online Retail (UCI)

**Algorithm:** Apriori / FP-Growth

**Evaluation metrics:** Support, Confidence, Lift

---

## Pipeline

Each task includes the following steps:

1. **Data loading** from the chosen dataset
2. **Preprocessing**
   - Removing duplicates
   - Handling missing values
   - Outlier detection
   - Encoding independent and dependent variables
   - Feature scaling
   - Handling class imbalance (classification)
   - Feature selection
3. **Train/Test split** (or cross-validation)
4. **Model training**
5. **Predicting new values**
6. **Visualization** of predicted vs. actual results and supporting plots
7. **Evaluation** using the metrics listed above

## Repository Structure

```
├── classification/     # Network Intrusion Detection
├── regression/         # Student Performance Prediction
├── association_rules/  # Product Recommendation
├── data/               # Datasets
└── README.md
```

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- mlxtend (Apriori / FP-Growth)
- Jupyter Notebook / Google Colab

## Installation

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install pandas numpy scikit-learn matplotlib seaborn mlxtend jupyter
```

## Team Members

- Member 1
- Member 2
- Member 3
- Member 4
- Member 5
- Member 6
- Member 7

## Course

Data Mining, Faculty/University Name, Fall 2026
