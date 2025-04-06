# Predictive-Maintenance-System---Using-AI-and-Data-Science


## Project Summery:

This project demonstrates the development of a predictive maintenance system using the AI4I 2020 dataset. It applies machine learning techniques to predict potential machine failures based on sensor readings, with the goal of reducing downtime and optimizing maintenance schedules.

## Dataset Description:

- **Source**: [AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance)
- **Samples**: 10,000
- **Features**:
  - Machine Type (L, M, H)
  - Air Temperature, Process Temperature
  - Rotational Speed, Torque, Tool Wear
  - **Target**: Machine Failure (0 = No, 1 = Yes)

## Objective:
Build a model to predict machine failure and:
  - Reduce unplanned downtime
  - Improve equipment effectiveness
  - Optimize maintenance schedules

##  Technologies Used:

- Google Colab Notebook or Jupyter Notebook
- Python
- Pandas, NumPy, Seaborn, Matplotlib
- Scikit-learn
- Imbalanced-learn (SMOTE)
- XGBoost & LightGBM
- Keras (Neural Networks)
- Joblib (Model Saving)

## Steps & Approach

1. **Data Preprocessing**
   - Handled missing values (if any)
   - Standardized features with `StandardScaler`
   - Balanced target variable using `SMOTE`

2. **Model Training**
   - Trained and evaluated multiple classifiers:
     - ✅ Random Forest
     - ✅ Support Vector Machine (SVM)
     - ✅ XGBoost
     - ✅ LightGBM
     - ✅ Neural Network (Keras)

3. **Model Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-Score
   - Used classification report and validation plots

4. **Model Selection**
   - Random Forest performed the best with **Accuracy: 0.9990**
   - Saved best model using `joblib`

## Results:

| Model         | Accuracy | F1 Score |
|---------------|----------|----------|
| Random Forest | 0.9990   | 0.9833   |
| SVM           | 0.9858   | 0.9856   |
| XGBoost       | 0.9959   | 0.9958   |
| LightGBM      | 0.9948   | 0.9948   |

## How to Run:

```bash
git clone https://github.com/sujata1712/Predictive-Maintenance-System---Using-AI-and-Data-Science.git
cd Predictive-Maintenance-System---Using-AI-and-Data-Science
pip install -r requirements.txt



