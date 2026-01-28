# Customer Churn Prediction using Logistic Regression

## 📌 Project Overview
This project predicts customer churn using Logistic Regression to help businesses 
identify customers at risk of leaving and take proactive actions.

## 📂 Project Structure
- DataSet (Raw data): Original dataset used for training
- Data analysis: Exploratory Data Analysis and preprocessing
- Logistic Regression: Model training, evaluation, and tuning
- Model: Saved trained model (joblib)

## 🛠️ Tools & Technologies
Python, Pandas, Scikit-learn, Matplotlib, Seaborn

## 📊 Model Performance
- Accuracy: 74%
- Churn Recall: 78%
- Improved recall using class imbalance handling

## 🚀 How to Run
1. Clone the repository
2. Run the notebooks in order
3. Load the saved model and test predictions


____________________________________________________________________________________________
Due to class imbalance, recall for churn customers was prioritized over accuracy
to better align with business objectives.

AR:

التنبؤ بمغادرة العملاء باستخدام الانحدار اللوجستي
📌 نظرة عامة على المشروع

يهدف هذا المشروع إلى التنبؤ باحتمالية مغادرة العملاء باستخدام خوارزمية  Logistic Regression
لمساعدة الشركات على تحديد العملاء المعرّضين لترك الخدمة       
واتخاذ إجراءات استباقية للحفاظ عليهم.

📂 هيكل المشروع

                                                                                                                                                         DataSet (Raw data): البيانات الأصلية المستخدمة في التدريب

                                                                                                                                                      Data analysis: تحليل البيانات الاستكشافي (EDA) ومعالجة البيانات

                                                                                                                                                             Logistic Regression: تدريب النموذج، تقييمه، وضبطه

                                                                                                                                                                  Model: النموذج المدرّب والمحفوظ باستخدام joblib

🛠️ الأدوات والتقنيات المستخدمة

                                                                                                                                                       Python، Pandas، Scikit-learn، Matplotlib، Seaborn

📊 أداء النموذج

الدقة (Accuracy): ‎74%

معدل الاستدعاء لمغادرة العملاء (Churn Recall): ‎78%

تم تحسين قدرة النموذج على اكتشاف العملاء المعرضين للمغادرة من خلال التعامل مع عدم توازن البيانات

🚀 كيفية تشغيل المشروع

نسخ المستودع (Clone repository)

تشغيل دفاتر Jupyter بالترتيب

تحميل النموذج المحفوظ واختبار التنبؤات
                                                                                                            _______________________________________________________________________________________

بسبب عدم توازن الفئات، تم إعطاء الأولوية لاسترجاع بيانات العملاء المتسربين على حساب الدقة لتحقيق توافق أفضل مع أهداف العمل
