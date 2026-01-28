# =========================
# 1️⃣ Import Libraries
# =========================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from joblib import dump, load

# =========================
# 2️⃣ Load Dataset
# =========================
df = pd.read_csv(
   r"C:\Users\admin\Desktop\logstic-Ew-pt\DataSet(Raw data)\WA_Fn-UseC_-Telco-Customer-Churn999.csv"
)

print(df.info())
print(df.isnull().sum())
print(df.head())



# =========================
# 3️⃣ Split Features & Target
# =========================
X = df.drop("Churn", axis=1)
y = df["Churn"]

# =========================
# 4️⃣ Train Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 5️⃣ Build Pipeline (Best Practice)
# =========================
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=1000,
        penalty="l2",
        C=1.0,
        class_weight="balanced",

        
    ))
])

# =========================
# 6️⃣ Train Model
# =========================
pipeline.fit(X_train, y_train)

# =========================
# 7️⃣ Predictions
# =========================
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

# =========================
# 8️⃣ Evaluation
# =========================
print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_test_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# =========================
# 9️⃣ Save Model
# =========================
dump(pipeline, "churn_logistic_pipeline.joblib")
print("✅ Model saved successfully")

# =========================
# 🔟 Load Model (Test)
# =========================
loaded_model = load("churn_logistic_pipeline.joblib")
print("✅ Model loaded successfully")

# =========================
# 1️⃣1️⃣ Test Loaded Model
# =========================
y_loaded_pred = loaded_model.predict(X_test)
print("Loaded Model Accuracy:", accuracy_score(y_test, y_loaded_pred))

# =========================
# 1️⃣2️⃣ Predict New Customer
# =========================
new_customer = X_test.iloc[[0]]  # مثال عميل جديد

prediction = loaded_model.predict(new_customer)
probability = loaded_model.predict_proba(new_customer)

print("Churn Prediction:", prediction[0])
print("Churn Probability:", probability[0][1])

# =========================
# 7️⃣ التقييم
# =========================


#cm=confusion_matrix(y_test,y_train_pred)
#ConfusionMatrixDisplay(cm,display_labels=model.classes_).plot(cmap='Blues')
#plt.title("confusion_matrix")
#plt.show() 

