# Diabetes Prediction Machine Learning Pipeline

## Introduction

This project aims to develop a robust machine learning model for predicting whether people have diabetes based on their characteristics. The dataset used for training and evaluation is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. The dataset consists of medical predictor variables such as the number of pregnancies, glucose concentration, blood pressure, skin thickness, insulin level, BMI, diabetes pedigree function, and age. The target variable is the presence or absence of diabetes.

## Business Problem

The ability to accurately predict diabetes in individuals can greatly assist in early diagnosis and treatment, leading to improved healthcare outcomes. By developing a machine learning model, we can provide a valuable tool for healthcare professionals to assess the likelihood of diabetes based on patient characteristics. This can help in targeted interventions and personalized healthcare plans.

## Dataset

The dataset used in this project contains 768 observation units and 9 variables. The variables in the dataset are as follows:

| Feature                  | Definition                                                               |
| ------------------------ | ------------------------------------------------------------------------ |
| Pregnancies              | Number of times pregnant                                                 |
| Glucose                  | Plasma glucose concentration a 2 hours in an oral glucose tolerance test |
| BloodPressure            | Diastolic blood pressure (mm Hg)                                         |
| SkinThickness            | Triceps skin fold thickness (mm)                                         |
| Insulin                  | 2-Hour serum insulin (mu U/ml)                                           |
| BMI                      | Body mass index (weight in kg/(height in m)^2)                           |
| DiabetesPedigreeFunction | Diabetes pedigree function                                               |
| Age                      | Age (years)                                                              |
| Outcome                  | Class variable (0 - non-diabetic, 1 - diabetic)                          |

## Usage

### Script Modes

- `--no-debug`: Runs the full script.
- `--no-tuning`: Runs the script without tuning.
- `--model-history`: Shows models parameters and RMSE scores.

### Make Commands

To run the script using make commands, first install the MakeFile application in your IDE, and then execute the following commands:

- `run`: Runs the full script with tuning.
- `debug`: Runs the script in debug mode (set the number of rows option in the `config.file`).
- `notuning`: Runs the script without tuning.
- `req`: Creates the `requirements.txt` file.
- `install`: Installs the packages listed in `requirements.txt`.
- `models`: Shows models parameters and RMSE scores.

## Files

- [_diabetes-prediction.ipynb_](https://github.com/oguzerdo/diabetes-prediction/blob/main/diabetes_prediction.ipynb): Project Notebook
- [_main.py_](https://github.com/oguzerdo/diabetes-prediction-ml-pipeline/blob/main/main.py): Main script
- [_configs.py_](https://github.com/oguzerdo/diabetes-prediction-ml-pipeline/blob/main/scripts/config.py): Configuration Files (Grid & Project settings)
- [_model_history.py_](https://github.com/oguzerdo/diabetes-prediction/blob/main/helpers.py): Show model validation scores from model_info_data.json file
- [_preprocess.py_](https://github.com/oguzerdo/diabetes-prediction-ml-pipeline/blob/main/scripts/preprocess.py): Data Preparation script
- [_train.py_](https://github.com/oguzerdo/diabetes-prediction-ml-pipeline/blob/main/scripts/train.py): Model Training with Debug option
- [_utils.py_](https://github.com/oguzerdo/diabetes-prediction-ml-pipeline/blob/main/scripts/utils.py): Helper functions
- [_outputs_](https://github.com/oguzerdo/diabetes-prediction-ml-pipeline/tree/main/outputs): Output files, includes model pkl objects and model validation history

## Requirements

The following Python packages are required to run the project:

```
joblib==1.1.0
lightgbm==3.1.1
matplotlib==3.5.2
numpy==1.22.3
pandas==1.4.4
scikit_learn==1.1.2
seaborn==0.11.2
xgboost==1.5.0
```
