# student-placement-salary-prediction
Machine learning project for predicting student placement status and salary using classification and regression models.

# Student Placement & Salary Prediction

## Background

Student employability can be influenced by various factors, including academic performance, technical skills, practical experience, and other student characteristics. Understanding how these factors relate to placement outcomes can help identify patterns associated with students' career readiness.

This project develops an end-to-end machine learning solution to analyze student placement outcomes through two predictive tasks. The first task is a **classification problem** that predicts whether a student will be placed, while the second is a **regression problem** that predicts the expected salary in LPA (Lakhs Per Annum).

In addition to predictive modeling, the project includes exploratory data analysis, statistical association analysis, feature engineering, feature selection, class imbalance handling, hyperparameter tuning, model evaluation, and deployment preparation.


## Objectives

The main objectives of this project are:

- Analyze academic, skill, experience, and other student-related factors associated with placement outcomes.
- Build a classification model to predict student placement status.
- Build a regression model to estimate salary in LPA.
- Perform feature selection using correlation analysis and statistical tests.
- Engineer additional features that better represent students' academic performance, skills, and practical experience.
- Address class imbalance in the placement classification task.
- Compare multiple machine learning algorithms and select suitable models based on evaluation metrics.
- Prepare the selected models for deployment through a prediction application and API.


## Dataset

The dataset contains student-level information covering several aspects of academic performance, skills, experience, lifestyle, and placement outcomes.

Some of the main variables include:

### Academic
- `cgpa`
- `tenth_percentage`
- `twelfth_percentage`
- `backlogs`
- `attendance_percentage`

### Skills & Experience
- `coding_skill_rating`
- `communication_skill_rating`
- `aptitude_skill_rating`
- `projects_completed`
- `internships_completed`
- `hackathons_participated`
- `certifications_count`

### Student Characteristics
- `gender`
- `branch`
- `part_time_job`
- `family_income_level`
- `city_tier`
- `internet_access`
- `extracurricular_involvement`

### Targets
- `placement_status` — classification target
- `salary_lpa` — regression target

The placement target is imbalanced, with approximately **86% of students placed and 14% not placed**, which requires additional consideration during classification modeling.


## Project Workflow

The project follows the following machine learning workflow:

1. Data loading and initial inspection
2. Missing value handling
3. Exploratory Data Analysis (EDA)
4. Target analysis
5. Numerical and categorical feature analysis
6. Correlation and statistical association analysis
7. Feature selection
8. Feature engineering
9. Classification and regression dataset preparation
10. Train-test splitting
11. Data preprocessing
12. Model training and cross-validation
13. Class imbalance handling for classification
14. Hyperparameter tuning
15. Model evaluation and selection
16. Model deployment preparation


## Exploratory Data Analysis

EDA was conducted separately for the classification and regression objectives.

For the placement classification task, several academic and experience-related variables showed noticeable differences between placed and non-placed students. Higher CGPA, project participation, internship experience, coding skills, and aptitude ratings were generally associated with placed students, while higher backlog counts showed a negative relationship with placement outcomes.

Several lifestyle and demographic variables showed weaker relationships with placement status.

For salary prediction, academic performance and practical experience showed more meaningful relationships with salary, while several lifestyle-related variables demonstrated relatively weak relationships.

Missing values in `extracurricular_involvement` were handled by introducing an `Unknown` category instead of dropping the affected observations.


## Feature Selection

Feature selection was performed separately for numerical and categorical variables.

Numerical relationships were evaluated using correlation analysis, while categorical variables were evaluated using statistical association analysis such as **Chi-Square tests and Cramer's V**.

Features showing weak relationships with the corresponding prediction target were considered for removal. Target-derived variables were also excluded where necessary to prevent **data leakage**.

For example, `salary_lpa` was excluded from the placement classification feature set because salary is directly related to the placement outcome.


## Feature Engineering

Several new features were created by combining related information from the original dataset.

The engineered features represent broader aspects of student performance:

- `academic_score` — summarizes academic-related information.
- `total_experience` — summarizes practical experience such as projects, internships, and hackathon participation.
- `total_skills` — summarizes relevant student skill ratings.

These engineered features were introduced to provide more compact representations of related characteristics and help the models capture broader patterns in the dataset.


# Machine Learning

## 1. Placement Classification

The classification task predicts whether a student will be **Placed** or **Not Placed**.

Three classification algorithms were compared:

- Logistic Regression
- Random Forest
- XGBoost

Because the target distribution is imbalanced, two imbalance-handling approaches were evaluated:

- Class weighting
- SMOTE (Synthetic Minority Over-sampling Technique)

Five-fold stratified cross-validation was used to compare model performance using:

- Accuracy
- Precision
- Recall
- F1-Score

Class weighting was selected after comparing its performance with SMOTE, particularly considering recall and F1-score rather than relying only on accuracy.

### Classification Evaluation

The final classification models were further evaluated using:

- Confusion Matrix
- Classification Report
- ROC-AUC

Logistic Regression achieved a **ROC-AUC of approximately 0.914** and was selected as the final classification model.

Despite its relatively simple structure, Logistic Regression performed competitively against the more complex tree-based models, suggesting that increasing model complexity did not provide a substantial improvement for this dataset.


## 2. Salary Regression

The regression task predicts `salary_lpa`.

The initial regression experiments compared:

- Linear Regression
- Ridge Regression
- Lasso Regression

Five-fold cross-validation was performed using:

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

Hyperparameter optimization was subsequently performed using **Optuna**.

### Regression Evaluation

The tuned models produced relatively similar results. The final **Lasso Regression** model achieved approximately:

- **RMSE: 4.08 LPA**
- **R²: 0.57**

Lasso Regression was selected as the final regression model because it achieved the lowest RMSE and highest R² among the evaluated regression models while also providing regularization and automatic coefficient shrinkage.

The regression results indicate that the available features explain part, but not all, of the variation in student salary, suggesting that additional factors not represented in the dataset may also contribute to salary outcomes.


## Key Findings

Several findings emerged from the analysis:

- Academic performance, particularly CGPA, is associated with placement outcomes.
- Practical experience through projects, internships, and hackathons provides useful predictive information.
- Coding and aptitude skills show stronger relationships with placement compared with several lifestyle-related variables.
- Higher backlog counts are negatively associated with placement outcomes.
- Several demographic and lifestyle variables showed relatively weak associations with the prediction targets.
- More complex machine learning models did not necessarily outperform simpler linear models in this dataset.
- Salary prediction remains more challenging than placement classification based on the available features.

## Model Evaluation & Results

### Classification — Placement Prediction

The classification task aims to predict whether a student will be placed or not. Since the target variable is imbalanced, model evaluation was not based solely on accuracy. Precision, recall, F1-score, and ROC-AUC were also considered.

Three classification algorithms were evaluated:

- Logistic Regression
- Random Forest
- XGBoost

To address class imbalance, both **class weighting** and **SMOTE** were evaluated using 5-fold Stratified Cross-Validation.

#### Class Weight Results

| Model | Accuracy | Precision | Recall | F1-Score |

| Logistic Regression | 0.8216 | 0.9689 | 0.8190 | 0.8876 |

| Random Forest | 0.8808 | 0.8985 | 0.9712 | 0.9334 |

| XGBoost | 0.8538 | 0.9402 | 0.8866 | 0.9125 |

Random Forest achieved the highest F1-score and recall under the class-weight approach, while Logistic Regression achieved the highest precision.

The results demonstrate why accuracy alone is insufficient for this dataset. Considering recall and F1-score provides a more informative assessment of model performance under class imbalance.

### Final Classification Model

After comparing imbalance-handling strategies and model performance, the selected classification model was evaluated on the held-out test set.

The final evaluation included:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC

The final model demonstrated strong ability to distinguish between placed and non-placed students. The confusion matrix was also analyzed to understand the types of classification errors rather than relying solely on overall accuracy.

## Salary Prediction — Regression

The regression task aims to predict student salary in LPA.

Three linear regression approaches were initially compared using 5-fold cross-validation:

| Model | RMSE | MAE | R² |

| Linear Regression | 4.0518 | 2.8559 | 0.5790 |

| Ridge Regression | 4.0518 | 2.8558 | 0.5790 |

| Lasso Regression | **4.0510** | **2.8509** | **0.5792** |

Lower RMSE and MAE indicate better prediction accuracy, while a higher R² indicates that the model explains a greater proportion of salary variation.

The baseline results show that all three linear models produced very similar performance, with Lasso achieving a slightly lower prediction error.

### Hyperparameter Optimization

Optuna was used to optimize the regression models using RMSE as the optimization objective.

The tuning process evaluated:

- `fit_intercept` for Linear Regression
- `alpha` for Ridge Regression
- `alpha` for Lasso Regression

After hyperparameter optimization, Lasso Regression was selected as the final regression model.

### Final Regression Result

The selected Lasso model achieved approximately:

- **RMSE: 4.08 LPA**
- **R²: 0.57**

An RMSE of approximately 4.08 indicates that the model's salary predictions differ from the actual salary by roughly 4.08 LPA based on the RMSE metric.

The R² score indicates that approximately 57% of the variation in salary can be explained by the features included in the model.

Although the differences between the three regression models were relatively small, Lasso provided the strongest evaluated performance while also introducing regularization through coefficient shrinkage.

## Project Highlights

- Developed two ML tasks: student placement classification and salary regression.
- Addressed class imbalance using Class Weight and SMOTE.
- Compared Logistic Regression, Random Forest, and XGBoost for classification.
- Compared Linear Regression, Ridge, and Lasso for salary prediction.
- Applied 5-fold cross-validation and Optuna hyperparameter optimization.
- Final regression model achieved approximately **RMSE 4.08 LPA and R² 0.57**.
- Prepared trained models for deployment through prediction applications and APIs.

  Streamlit Public Link : https://mid-exam-project-finalized-classification.streamlit.app/
