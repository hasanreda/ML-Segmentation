# Machine Learning Driven User Segmentation

## Objective

The objective of this project is to construct a machine learning model that accurately classifies entities into predefined segments, learning from an existing labeled dataset. The steps to achieve this objective include:

1. Data Exploration: Investigate the dataset to understand the feature distributions, handle missing values, and explore feature interrelationships.

2. Preprocessing: Prepare the data for modeling by imputing missing values, encoding categorical variables, normalizing numerical inputs, and addressing outliers.

3. Feature Engineering: Enhance the model's predictive power by selecting and possibly constructing key features that have the most significant impact on classification.

4. Model Development: Identify and test various classification algorithms suitable for the data's characteristics.

5. Model Training and Validation: Train the models on the dataset and validate their performance with metrics like accuracy and F1-score, employing cross-validation for robustness.

6. Hyperparameter Optimization: Refine the models by tuning hyperparameters to improve accuracy and reduce overfitting.

7. Model Selection: Evaluate and select the optimal model based on performance metrics, balancing predictive power with computational efficiency.

8. Model Deployment: Implement the final model to classify new data, aligning its predictive segmentation with the established categories.

The goal is to leverage the existing segmentation framework to inform the model, ensuring that it not only predicts accurately but also aligns with the established understanding of the segment structures.

## Overview:

The dataset comprises 10,695 records and consists of 10 columns. Unfortunately, the features lack detailed descriptions as provided by the data source.

Feature Variables:

    Discrete Features: Age, Work_Experience, Family_Size

    Boolean Features: Gender, Ever_Married, Graduated

    Nominal Features: Profession, Var_1

    Ordinal Features: Spending_Score

Target Label:

    Segmentation



## Models

The following machine learning models are used in this project:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- XGBoost

Each model's performance is evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score

## Usage

To run the project, execute the Jupyter notebook `Customer segmentation.ipynb`.

## Results

The performance of each model is stored in a pandas DataFrame `df_models`. This DataFrame contains the accuracy, precision, recall, and F1 score for each model.

Based on the error metrics calculated for each model, it appears that none of the four models achieved an accuracy score higher than 50%, although it is much better than a coin flip, which would be of 25% accuracy, This could be due to a number of factors, such as the size and quality of the dataset, the complexity of the problem being solved, or the choice of features and hyperparameters used in each model.

XGBoost performed the best among the models, with the highest accuracy, precision, recall, and F1 score. Therefore, XGBoost would be the recommended model to use for now. It is important to note that further iterations and refinements may be necessary to improve the performance of the models.

## Dependencies

This project requires the following Python libraries:

- pandas
- sklearn
- xgboost

