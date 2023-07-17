import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

# Load the Decision Tree Bagging model from the joblib file
model = joblib.load('TCHC_CV_model/DTB_model.joblib')

# Load the testing data from the CSV file
test_data = pd.read_csv('T_for_AINA_2022.csv')

# Extract the features from the testing data
X_test = test_data.drop('CV', axis=1)  # Assuming 'label' column is the target variable

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Extract the actual labels from the testing data
y_true = test_data['CV']

# Compute the accuracy of the model
accuracy = accuracy_score(y_true, y_pred)

result = {
    "model_name":["DTB"],
    "Accuracy":[accuracy]
}
import os
if not os.path.exists("result"):
    os.mkdir("result")
result = pd.DataFrame(result)
result.to_csv("result/DTB_result.csv", index=False)

print("Accuracy:", accuracy)