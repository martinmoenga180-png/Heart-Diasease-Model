import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, ConfusionMatrixDisplay
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE

#Load dataset
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

#Check MVs
print("Missing Values per column:\n",train_df.isna().sum())

#Outlier Check
numeric_cols = train_df.select_dtypes(include="number").columns
Q1, Q3 = train_df[numeric_cols].quantile(0.25), train_df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
outlier_mask = (train_df[numeric_cols] < lower) | (train_df[numeric_cols] > upper)
train_df["has_outlier"] = outlier_mask.any(axis=1)
df_clean = train_df[~train_df["has_outlier"]]

#Encoding 
df_new = df_clean.copy()
df_new = df_new.drop(columns=["id","has_outlier"], axis=1)
df_new["Heart Disease"] = df_new["Heart Disease"].map({"Presence":1, "Absence":0})

#Multi-collinearity 
x = df_new.drop(columns=["Heart Disease"], axis=1)
y = df_new["Heart Disease"]
corr_matrix = x.corr()
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i,j]) > 0.8:
            high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i,j]))

features_to_drop = [f2 for f1, f2, corr in high_corr]
x_final = x.drop(columns=features_to_drop, errors='ignore')

#Split for Evaluation
x_train_sub, x_val, y_train_sub, y_val = train_test_split(x_final, y, test_size=0.2, stratify=y, random_state=42)

#Models
models = {
    "LogisticRegression": LogisticRegression(class_weight='balanced', max_iter=5000, random_state=42),
    "SVM": SVC(class_weight='balanced', probability=True, random_state=42),
    "DecisionTree": DecisionTreeClassifier(class_weight='balanced', random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=12, class_weight='balanced', random_state=42)
}

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#THE LOOP
for name, model in models.items():
    print(f"\nProcessing {name}...")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("variance", VarianceThreshold()),
        ("model", model)
    ])

    #param grids
    if name == "LogisticRegression":
        param_grid = {'variance__threshold':[0,0.01], 'model__C':[0.1, 1, 10], 'model__solver':['lbfgs','liblinear'], 'model__penalty': ['l2']}
    elif name == "SVM":
        param_grid = {'variance__threshold':[0,0.01], 'model__C':[0.1, 1, 10], 'model__kernel':['linear', 'rbf']}
    elif name == "DecisionTree":
        param_grid = {'variance__threshold':[0,0.01], 'model__max_depth':[3,5,7], 'model__criterion':['gini','entropy']}
    elif name == "RandomForest":
        param_grid = {'variance__threshold':[0,0.01], 'model__n_estimators':[100, 200], 'model__max_depth':[3,5,7]}

    grid = GridSearchCV(pipe, param_grid=param_grid, scoring='accuracy', cv=kf, n_jobs=-1)
    grid.fit(x_train_sub, y_train_sub)

    #EVALUATION 
    best_model = grid.best_estimator_
    y_pred_val = best_model.predict(x_val)

    print(f"--- {name} Classification Report ---")
    print(classification_report(y_val, y_pred_val))
    print(f"Accuracy: {accuracy_score(y_val, y_pred_val):.4f}")

    #Confusion Matrix
    plt.figure(figsize=(5,4))
    sns.heatmap(confusion_matrix(y_val, y_pred_val), annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix: {name}")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    #Learning Curve
    train_sizes, train_scores, test_scores = learning_curve(best_model, x_train_sub, y_train_sub, cv=kf, scoring='accuracy', n_jobs=-1)
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training accuracy')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Validation accuracy')
    plt.title(f"Learning Curve: {name}")
    plt.legend()
    plt.show()

#Final Predict & Save
x_test_final = test_df.drop(columns=["id"], errors='ignore')[x_final.columns]
y_pred_final = best_model.predict(x_test_final)
output = pd.DataFrame({"id": test_df["id"], "target": y_pred_final})
output.to_csv("Predictions.csv", index=False)