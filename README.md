**Breast Cancer Classification – Machine Learning Assignment 2**

**1. Problem Statement**

Breast cancer classification is an important machine learning application where tumor characteristics can be used to classify a diagnosis as Benign or Malignant.

The objective of this project is to implement and compare multiple machine learning classification algorithms using the Breast Cancer (UCI) dataset. The models are evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

An interactive Streamlit web application was also developed to demonstrate model performance and perform predictions on test data.


**2. Dataset Description**

The Breast Cancer (UCI) dataset used in this project contains:

569 instances
30 input features
Binary classification target
455 training samples
114 testing samples

The target variable is:

0 = Benign
1 = Malignant

The dataset contains numerical features describing characteristics of breast cell nuclei, including measurements such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.

The dataset satisfies the assignment requirements of a minimum of 500 instances and 12 features.

3. GitHub Repository Link

      GitHub Repository:

  https://github.com/Tanushree-Samanta1/breast-cancer-classification-ml

4. Machine Learning Models Used

The following five classification models were implemented on the same dataset:

Logistic Regression
Decision Tree Classifier
K-Nearest Neighbors (kNN)
Gaussian Naive Bayes
Random Forest Classifier (Ensemble)
Evaluation Metrics

Each model was evaluated using:

Accuracy
AUC Score
Precision
Recall
F1 Score
Matthews Correlation Coefficient (MCC)

| ML Model Name            |   Accuracy |        AUC |  Precision |     Recall |   F1 Score |        MCC |
| ------------------------ | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| **Logistic Regression**  | **0.9825** | **0.9954** | **0.9861** | **0.9861** | **0.9861** | **0.9623** |
| Decision Tree            |     0.9123 |     0.9157 |     0.9559 |     0.9028 |     0.9286 |     0.8174 |
| kNN                      |     0.9561 |     0.9788 |     0.9589 |     0.9722 |     0.9655 |     0.9054 |
| Naive Bayes              |     0.9298 |     0.9868 |     0.9444 |     0.9444 |     0.9444 |     0.8492 |
| Random Forest (Ensemble) |     0.9561 |     0.9937 |     0.9589 |     0.9722 |     0.9655 |     0.9054 |


5. Model Performance Observations
Logistic Regression

Logistic Regression achieved the best overall performance among all the models. It achieved an Accuracy of 98.25% and an AUC of 99.54%. It also achieved the highest Precision, Recall, F1 Score, and MCC. This indicates highly accurate and balanced classification performance on the selected dataset.

Decision Tree

The Decision Tree achieved an Accuracy of 91.23% and an AUC of 91.57%. It had the lowest overall performance among the five models, particularly in terms of Accuracy, Recall, F1 Score, and MCC.

kNN

The kNN model achieved an Accuracy of 95.61% and an AUC of 97.88%. Its Recall of 97.22% and F1 Score of 96.55% indicate strong classification performance. However, it performed below Logistic Regression overall.

Naive Bayes

Naive Bayes achieved an Accuracy of 92.98% and an AUC of 98.68%. Although its AUC was high, its Accuracy, Precision, Recall, F1 Score, and MCC were lower than the corresponding results of Logistic Regression, kNN, and Random Forest.

Random Forest (Ensemble)

Random Forest achieved an Accuracy of 95.61% and an AUC of 99.37%. It demonstrated strong classification performance and an AUC very close to Logistic Regression. However, Logistic Regression performed better across the overall set of evaluation metrics.

6. Overall Winner

Logistic Regression

Logistic Regression was selected as the overall best-performing model for this dataset.

Its performance was:
| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **98.25%** |
| AUC       | **99.54%** |
| Precision | **98.61%** |
| Recall    | **98.61%** |
| F1 Score  | **98.61%** |
| MCC       | **96.23%** |

The model achieved the highest values across all six evaluation metrics in the comparison.


7. Streamlit Web Application

An interactive Streamlit application was developed to demonstrate the machine learning models.

Application Features

The application provides:

Dataset overview
Test-data CSV upload
Model performance comparison
Display of all six evaluation metrics
Best-performing model identification
Confusion matrix
Classification report
Machine learning model selection
Test sample selection
Breast cancer prediction
Malignant probability display
Correct/incorrect prediction indication


Prediction Demonstration

A test prediction was performed using Logistic Regression on Test Sample 1.

The application produced:

Prediction: Malignant
Actual: Malignant
Malignant Probability: 100.00%
Prediction Status: Correct

8. Project Structure
   
breast-cancer-classification-ml/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
└── models/
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── logistic_regression.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── scaler.pkl
    
9. Technologies Used
   
Python
Pandas
NumPy
Scikit-learn
Streamlit
Matplotlib
Seaborn
Joblib

10. Live Streamlit Application

Streamlit App:

   https://breast-cancer-classification-ml-5ruzoxxhkdtcmqzpthwv4g.streamlit.app/



11. Conclusion

Five machine learning classification models were implemented and evaluated using the Breast Cancer (UCI) dataset.

Among the models tested, Logistic Regression achieved the best overall performance, with an Accuracy of 98.25%, AUC of 99.54%, Precision of 98.61%, Recall of 98.61%, F1 Score of 98.61%, and MCC of 96.23%.

The developed Streamlit application provides an interactive interface for exploring the dataset, comparing model performance, evaluating test data, and generating breast cancer predictions.



