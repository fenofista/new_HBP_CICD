import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score

# Load the XGBoost model from the bst file
model = xgb.Booster()
model.load_model('C:/Users/oplab/Desktop/new_HBP_CICD/TCHC_CV_model/XGB_model.bst')

# Load the testing data from the CSV file
test_data = pd.read_csv('C:/Users/oplab/Desktop/new_HBP_CICD/user_data/T_for_AINA_2022.csv')

# Extract the features from the testing data
X_test = test_data.drop('CV', axis=1)  # Assuming 'label' column is the target variable

# Make predictions on the testing data
y_pred = model.predict(xgb.DMatrix(X_test))

# Convert the predicted values to binary labels (if necessary)
y_pred = [1 if pred > 0.5 else 0 for pred in y_pred]

# Extract the actual labels from the testing data
y_true = test_data['CV']

# Compute the accuracy of the model
accuracy = accuracy_score(y_true, y_pred)
id_arr=[]
for i in range(len(y_pred)):
    id_arr.append(i)




result = {
    "patient_id":id_arr,
    "pred":y_pred
}
import os
if not os.path.exists("C:/Users/oplab/Desktop/new_HBP_CICD/result"):
    os.mkdir("C:/Users/oplab/Desktop/new_HBP_CICD/result")
result = pd.DataFrame(result)
result.to_csv("C:/Users/oplab/Desktop/new_HBP_CICD/result/XGB_result.csv", index=False)

print("Accuracy:", accuracy)