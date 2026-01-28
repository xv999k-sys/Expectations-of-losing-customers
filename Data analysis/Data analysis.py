
import pandas as pd
import matplotlib.pyplot as plt

data2 = pd.read_csv(r"C:\Users\admin\Desktop\logstic-Ew-pt\DataSet(Raw data)\WA_Fn-UseC_-Telco-Customer-Churn.csv")


data2.drop(columns=['customerID','TotalCharges'] ,inplace=True)


data2['Churn'] = data2['Churn'].map({'Yes':1, 'No':0})


binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
data2[binary_cols] = data2[binary_cols].replace({'Yes':1, 'No':0})

gender=['gender']
data2[gender]=data2[gender].replace({'Male':0, 'Female':1})


service_cols = ['MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
data2 = pd.get_dummies(data2, columns=service_cols, drop_first=True)


multi_cols = ['InternetService', 'PaymentMethod', 'Contract']
data2 = pd.get_dummies(data2, columns=multi_cols, drop_first=True)

data2['MonthlyCharges'] = data2['MonthlyCharges'].round().astype(int)

data2 = data2.astype(int)


data2.to_csv(r"C:\Users\admin\Desktop\logstic-Ew-pt\DataSet(Raw data)\WA_Fn-UseC_-Telco-Customer-Churn999.csv",index=False)
print("saved to scv")
