# ❤️🩺 Heart Disease Prediction & Model Interpretation



![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg) ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg) ![Pandas](https://img.shields.io/badge/Pandas-blueviolet.svg) ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-brightgreen.svg) ![Matplotlib](https://img.shields.io/badge/Matplotlib-grey.svg) ![Seaborn](https://img.shields.io/badge/Seaborn-darkblue.svg)

This project provides a comprehensive analysis of the **UCI Heart Disease dataset**. Beyond simply building predictive models, the core focus of this notebook is on **model interpretability**, utilizing advanced techniques like **Partial Dependence Plots (PDP)** and **Individual Conditional Expectation (ICE)** plots to understand *why* a model makes its predictions.

---

## CONTENTS

- [Overview](#overview)
- [Project Workflow](#workflow)
- [Key Analysis & Visualizations](#visualizations)
- [Models & Evaluation](#models)
- [Model Interpretation Deep Dive](#interpretation)
- [How to Use](#how-to-use)
- [Libraries Used](#libraries)

---

## <a name="overview"></a>📖 Overview

The primary goal is to predict the presence of heart disease (a binary classification task) based on 13 clinical features. This notebook provides an end-to-end walkthrough, from data acquisition and cleaning to model evaluation and, most importantly, interpretation.

### Key Objectives:
* Apply and compare `Logistic Regression` and `Random Forest` models.
* Evaluate model performance using classification reports and confusion matrices.
* Visualize feature importance (Random Forest) and model coefficients (Logistic Regression).
* **Deeply understand and interpret model behavior** using:
    * Partial Dependence Plots (PDP) for single and multiple features.
    * Individual Conditional Expectation (ICE) plots for instance-level analysis.

---

## <a name="workflow"></a>⚙️ Project Workflow

1.  **Data Acquisition:** Fetched the Heart Disease dataset directly from the `ucimlrepo` repository.
2.  **Data Preprocessing:**
    * Handled missing values (`ca`, `thal`) using `SimpleImputer` with a mean strategy.
    * Transformed the multi-class target variable (0-4) into a binary target (0: No Disease, 1: Disease Present).
3.  **Exploratory Data Analysis (EDA):**
    * Plotted feature distributions and generated a correlation matrix.
4.  **Modeling:**
    * Split the data into training and testing sets (80/20 split) using stratification.
    * Trained `LogisticRegression` and `RandomForestClassifier` models.
5.  **Model Interpretation:**
    * Analyzed and compared feature importance/coefficients.
    * Generated and analyzed 1D/2D PDPs and ICE plots to understand model logic.

---

## <a name="visualizations"></a>📊 Key Analysis & Visualizations

This notebook is rich with visualizations designed to build intuition about the data and the models.

### Correlation Heatmap
A heatmap helps to understand the relationships between different features and identify potential multicollinearity.



### Feature Distributions
Histograms and count plots show how features differ between healthy and sick patients, providing initial clues for which features are most predictive.

---

## <a name="models"></a>🎯 Models & Evaluation

Both models achieved an identical overall accuracy on the test set, but their internal logic (as revealed by the interpretation techniques) differs.

| Model | Accuracy | F1-Score (Class 1: Disease) |
| :--- | :---: | :---: |
| **Logistic Regression** | 84% | 0.79 |
| **Random Forest** | 84% | 0.80 |

### Model Performance
Confusion matrices provide a detailed breakdown of correct and incorrect predictions for each class.



---

## <a name="interpretation"></a>🧠 Model Interpretation Deep Dive

Accuracy isn't everything. This project's main value is in *opening the black box* to build trust and gain insights.

### Partial Dependence Plots (PDP)
We analyze how the model's prediction for heart disease changes, *on average*, as we vary a single feature. This plot for `age`, for example, shows a notable increase in predicted risk for patients in their mid-50s and above.



### 2D Partial Dependence
We explore the *interaction effect* between two features. The analysis of `age` and `thalach` (max heart rate) reveals complex relationships that a single-feature analysis would miss.

### Individual Conditional Expectation (ICE)
We go one level deeper than PDP to see how the prediction for *a single patient* changes. This helps uncover heterogeneous effects (individual differences) that are hidden by the PDP's average.



---

## <a name="how-to-use"></a>🚀 How to Use

1.  **Clone or download the repository.**
2.  **Install dependencies:**
    This project uses standard data science libraries.
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn ucimlrepo
    ```
3.  **Run the notebook:**
    ```bash
    jupyter notebook heart_disease.ipynb
    ```
4.  **(Optional) Add your own visuals:**
    * Run the notebook to generate your plots.
    * Save the plots (e.g., `plt.savefig('images/my_plot.png')`).
    * Update the Markdown links in this README file to point to your new images.

---

## <a name="libraries"></a>📚 Libraries Used

* [Numpy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Matplotlib](https://matplotlib.org/)
* [Seaborn](https://seaborn.pydata.org/)
* [Scikit-learn](https://scikit-learn.org/stable/)
* [ucimlrepo](https://pypi.org/project/ucimlrepo/) (for data fetching)
