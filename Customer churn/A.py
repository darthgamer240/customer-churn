#Importing Modules
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay,classification_report
import seaborn as sns
import shap

#Input
df = pd.read_csv('Customer churn\dataset.csv')

#Preprocessing
df.drop('customerID', axis=1, inplace=True)
df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df.dropna(inplace=True)
#One Hot Encoding
df=pd.get_dummies(df,dtype=int,columns=['InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','Contract','PaymentMethod'])
X = df.drop('Churn', axis=1)
y = df['Churn']
#Train and Test data split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)
#Scaling
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Random Forest Model
md=RandomForestClassifier(random_state=20)
md.fit(X_train,y_train)
y_pred_base=md.predict(X_test)

#RandomSearch
param_dist = {
    'n_estimators': [100, 300, 500, 800, 1200],
    'max_depth': [None, 5, 10, 15, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 6, 8],
    'bootstrap': [True, False],
    'class_weight': [None, 'balanced', 'balanced_subsample']
}
rsearch=RandomizedSearchCV(
    estimator=md,
    param_distributions=param_dist,
    n_iter=80,
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    scoring='recall'   
)
rsearch.fit(X_train, y_train)
md1 = rsearch.best_estimator_
y_pred=md1.predict(X_test)

#Stat Base
print("BASE:")
accuracy_base=accuracy_score(y_test,y_pred_base)
stat_base=classification_report(y_test,y_pred_base)
print("accuracy=",accuracy_base)
print(stat_base)

#Stat Tuned
print("TUNED:")
accuracy=accuracy_score(y_test,y_pred)
stat=classification_report(y_test,y_pred)
print("accuracy=",accuracy)
print(stat)
print("Best hyperparameters:", rsearch.best_params_)

#Confusion Matrix
cm=confusion_matrix(y_test,y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# Plot SHAP summary
explainer = shap.TreeExplainer(md1)
shap_values = explainer.shap_values(X_test)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
plt.title('SHAP Summary Plot - Features Impacting Churn')
plt.tight_layout()
plt.savefig('shap_summary.png')
plt.close()