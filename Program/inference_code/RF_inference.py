import pickle
import pandas as pd
from sklearn.metrics import accuracy_score

# Load the Random Forest model from the pkl file
with open('C:/Users/oplab/Desktop/new_HBP_CICD/models/RF_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the testing data from the CSV file
test_data = pd.read_csv('C:/Users/oplab/Desktop/new_HBP_CICD/user_data/T_for_AINA_2022_20230731.csv')

drop_index=[]
for index in test_data.index:
    try:
        tmp = float(test_data["temp"].loc[index])
        test_data["CV"].loc[index] = float(test_data["CV"].loc[index])-1
        if test_data["CV"].loc[index]>=2:
            drop_index.append(index)
    except:
        drop_index.append(index)
test_data = test_data.drop(drop_index)
test_data = test_data.reset_index()
test_data["temp"] = pd.to_numeric(test_data["temp"])
# Extract the features from the testing data
X_test = test_data.drop('CV', axis=1)  # Assuming 'label' column is the target variable
# Make predictions on the testing data
y_pred = model.predict(X_test)

# Extract the actual labels from the testing data
y_true = test_data['CV']
accuracy = accuracy_score(y_true, y_pred)
# Compute the accuracy of the model
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
result.to_csv("C:/Users/oplab/Desktop/new_HBP_CICD/result/RF_result.csv", index=False)

print("Accuracy:", accuracy)