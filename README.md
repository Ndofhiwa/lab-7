<div align="center">

# 🗳️ Machine Learning Lab 7 — Ensemble Methods

**One model is good. Five arguing models are better.**

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Ensembles-orange?style=flat-square&logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter)
![NumPy](https://img.shields.io/badge/NumPy-Arrays-013243?style=flat-square&logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-DataFrames-150458?style=flat-square&logo=pandas)

</div>

---

## 📋 Overview

`lab-7.ipynb` is a full deep-dive into **ensemble learning** — the family of techniques that combine multiple models to produce predictions stronger than any single classifier can achieve alone. The lab covers three core paradigms (voting, bagging, boosting), implements a custom classifier from scratch, and closes with a comprehensive comparison and tuning session across seven models.

| Part | Topic | Dataset |
|---|---|---|
| 1 | Majority Voting | Iris (binary subset) |
| 2 | Bagging | Wine (UCI) |
| 3 | AdaBoost | Wine (UCI) |
| 4 | Full Comparison + Tuning | Iris (full, 3-class) |

---

## 🗳️ Part 1 — Majority Voting Classifier

> *Democracy applied to machine learning: let the classifiers vote, and trust the majority.*

**Step 1.1 — Ensemble Concepts**
Introduces the core intuition behind ensemble methods: by combining multiple diverse classifiers, errors tend to cancel out while correct predictions reinforce each other — producing a model more robust than any individual learner.

**Step 1.2 — Custom `MajorityVoteClassifier` (Built from Scratch)**
Implements a fully functional majority vote ensemble classifier by subclassing `sklearn`'s `BaseEstimator` and `ClassifierMixin`. The class supports:
- `vote='classlabel'` — hard voting via `np.bincount` across all classifiers
- `vote='probability'` — soft voting by averaging predicted class probabilities
- Optional per-classifier `weights` for weighted voting
- `get_params()` override for compatibility with `GridSearchCV`
- Cloned base classifiers to prevent data leakage across fits

**Step 1.3 — Dataset Preparation**
Loads the Iris dataset and selects the two harder-to-separate classes (Versicolor and Virginica) using only sepal width and petal length features. Applies `LabelEncoder` and a stratified 70/30 train/test split.

**Step 1.4 — Training Individual Classifiers**
Trains three diverse base classifiers — each chosen to bring different inductive biases:
- **Logistic Regression** (L2, low C) — wrapped in a `StandardScaler` pipeline
- **Decision Tree** (max depth 1, entropy criterion) — a simple decision stump
- **KNN** (1 nearest neighbour, Euclidean distance) — wrapped in a `StandardScaler` pipeline

Evaluates all three using **10-fold cross-validation ROC AUC** as the scoring metric.

**Step 1.5 — Evaluating the Ensemble**
Combines the three classifiers into the custom `MajorityVoteClassifier` and re-evaluates it alongside the individuals using the same 10-fold CV setup. Demonstrates that the ensemble's mean ROC AUC exceeds each individual base classifier.

---

## 🎒 Part 2 — Bagging

> *Train 500 trees on slightly different slices of the data. Let them average each other out.*

**Step 2.1 — Bagging Concepts**
Explains Bootstrap Aggregating (Bagging): each base estimator is trained on a bootstrap sample (random draw with replacement) of the training data. Averaging across many such models reduces variance without increasing bias.

**Step 2.2 — Wine Dataset**
Loads the Wine dataset from the UCI repository, drops Class 1, and reduces to a binary classification task using only two features: `Alcohol` and `OD280/OD315 of diluted wines`. Applies `LabelEncoder` and an 80/20 stratified split.

**Step 2.3 — Bagging vs. Single Decision Tree**
Trains an unpruned `DecisionTreeClassifier` (full depth) as the base learner and compares it against a 500-estimator `BaggingClassifier` using the same base. Prints:
- Train and test accuracy for both models
- Accuracy improvement on training and test sets
- Reduction in the train/test gap — a direct measure of overfitting reduction

The unpruned tree massively overfits (near-perfect training accuracy, lower test accuracy). Bagging closes that gap significantly by averaging out the trees' idiosyncratic decisions.

**Step 2.4 — Decision Boundary Visualization**
Plots side-by-side decision boundaries for the single tree and the bagging ensemble on the 2D Wine feature space. The single tree shows jagged, overfit boundaries; the ensemble produces smoother, more generalized regions.

---

## 🚀 Part 3 — Adaptive Boosting (AdaBoost)

> *Train weak learners sequentially, each one fixing the mistakes of the last.*

**Step 3.1 — AdaBoost Concepts**
Introduces boosting as a sequential ensemble strategy: misclassified samples get higher weight in the next iteration, forcing each successive weak learner to focus on the hardest examples. Final predictions are a weighted vote of all learners.

**Step 3.2 — AdaBoost vs. Decision Stump**
Trains a single `DecisionTreeClassifier` with `max_depth=1` (a stump) and compares it against a 500-estimator `AdaBoostClassifier` wrapping the same stump at `learning_rate=0.1`. Prints:
- Train/test accuracy for both
- Accuracy improvement on both sets

A single stump is barely better than random chance on this dataset. AdaBoost's sequential ensemble transforms it into a high-accuracy classifier.

**Step 3.3 — Error Convergence Analysis**
Uses `staged_predict()` to track training and test error at every boosting iteration from 1 to 500. Plots both curves on the same axis. Demonstrates the characteristic pattern:
- Training error steadily decreases as the ensemble focuses on hard samples
- Test error follows a U-shape — improving initially, then creeping back up as the model begins to overfit training noise at high iteration counts

The plot helps identify the **optimal number of boosting rounds** before overfitting sets in.

---

## 📊 Part 4 — Full Comparison & Hyperparameter Tuning

> *Seven classifiers. One dataset. Ranked by cross-validated accuracy.*

**Step 4.1 — Comprehensive Comparison**
Returns to the full 3-class Iris dataset and evaluates all seven classifiers using 10-fold cross-validation accuracy:

| Classifier | Notes |
|---|---|
| Logistic Regression | Linear baseline |
| KNN | Instance-based baseline |
| Decision Tree | Single tree baseline |
| Random Forest | Bagging with feature randomness |
| Voting Classifier | Hard vote across all four base models |
| Bagging | Bootstrap aggregating on Decision Trees |
| AdaBoost | Sequential boosting on Decision Trees |

Results are collected into a `DataFrame` and sorted by mean accuracy, giving a clear performance ranking.

**Step 4.2 — GridSearchCV Tuning**
Applies 5-fold `GridSearchCV` to tune both ensemble methods:

- **AdaBoost** — searches over `n_estimators`, `learning_rate`, and `estimator__max_depth`
- **Bagging** — searches over `n_estimators`, `max_samples` (fraction of training data per bootstrap), and `max_features` (fraction of features per bootstrap)

Prints best parameter sets and best cross-validated scores for both.

---

## 💡 Key Insights

**When ensembles help and when they don't**
Majority voting gains strength from *diversity*. If the base classifiers are highly correlated or all biased in the same direction, errors don't cancel out — they amplify. Ensembles also struggle on very small datasets where bootstrap resampling leaves individual learners with too little data to learn from.

**Bagging and overfitting**
Bootstrap sampling forces each tree to learn from a slightly different data distribution, preventing any single tree from memorizing the full training set. Averaging 500 such trees smooths out erratic decision boundaries and significantly reduces the train/test gap seen in a single unpruned tree.

**AdaBoost and the learning rate**
A small learning rate (0.1) makes conservative weight updates, requiring more iterations but converging more stably. A large learning rate (1.0) converges faster but risks overshooting. The staged error plot reveals exactly when the model transitions from learning to overfitting.

**Scalability trade-offs**

| Method | Parallelizable | Sequential | Notes |
|---|---|---|---|
| Bagging / Random Forest | ✅ Yes | ❌ No | Linear scaling with estimators |
| AdaBoost / Boosting | ❌ No | ✅ Yes | Each tree depends on the previous |
| Voting / Stacking | ✅ Partial | ❌ No | Bottlenecked by slowest component |

**Real-world use cases covered:**
- Random Forest → financial fraud detection
- AdaBoost → medical diagnosis
- Gradient Boosting → predictive maintenance
- Bagging → real-time recommendation systems
- Voting / Stacking → competition / benchmark settings

---

## 🔧 Setup

```bash
pip install numpy pandas matplotlib scikit-learn
```

> Notebook developed and run in **Jupyter Notebook / JupyterLab**.

---

## 📊 Datasets Used

- **Iris Dataset** — 150 samples, 3 flower species, 4 features (Parts 1 & 4)
- **Wine Dataset (UCI)** — 178 samples, 3 cultivar classes, 13 chemical features (Parts 2 & 3)  
  Loaded from: `https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data`

---

<div align="center">
  <sub>Part of a Machine Learning coursework series · Python · scikit-learn · pandas</sub>
</div>
