import pandas as pd
import gc
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import roc_auc_score
from scipy.stats import uniform
import xgboost as xgb
import numpy as np


# Print all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# Load part of the train data
ctr15 = pd.read_csv("ctr_15.csv")
ctr16 = pd.read_csv("ctr_16.csv")
ctr17 = pd.read_csv("ctr_17.csv")
ctr18 = pd.read_csv("ctr_18.csv")
ctr19 = pd.read_csv("ctr_19.csv")
ctr20 = pd.read_csv("ctr_20.csv")
ctr21 = pd.read_csv("ctr_21.csv")
train_data = pd.concat([ctr15,ctr16,ctr17,ctr18,ctr19,ctr20,ctr21], ignore_index=True)

# Load the test data
test_data = pd.read_csv("ctr_test.csv")

# Train a tree on the train data
train_data = train_data.sample(frac=1/7, random_state=123)
y = train_data["Label"]
x = train_data.drop(columns=["Label"])
x = x.select_dtypes(include='number')
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=3456)
X_train, X_test_val, y_train, y_test_val = train_test_split(X_train, y_train, test_size=0.1, random_state=3456)
del train_data
gc.collect()


# Definir el espacio de búsqueda para Random Search
param_dist = {
    'max_depth': np.arange(5, 15, 1),
    'learning_rate': np.linspace(0.01, 0.3, 30),
    'n_estimators': np.arange(100, 1000, 50),
    'gamma': np.linspace(0, 0.5, 10),
    'min_child_weight': np.arange(10, 100, 10),
    'subsample': np.linspace(0.5, 1, 10),
    'colsample_bytree': np.linspace(0.5, 1, 10),
}

rs = RandomizedSearchCV(estimator = xgb.XGBClassifier(random_state = 22),
                        param_distributions = param_dist,
                        n_iter = 1000,
                        cv = KFold(4),
                        random_state = 22)

rs.fit(X_train, y_train)

# Imprimir los mejores parámetros encontrados
print("Mejores parámetros encontrados por Random Search:", rs.best_params_)

# Entrenar el modelo final con los mejores parámetros
clf_xgb_optimized = rs.best_estimator_

# Evaluar el modelo optimizado
preds_val_optimized = clf_xgb_optimized.predict_proba(X_test_val)[:, 1]
print("AUC test score - XGBoost Optimizado:", roc_auc_score(y_test_val, preds_val_optimized))


# Predict on the evaluation set
eval_data = test_data.select_dtypes(include='number')
y_preds = clf_xgb_optimized.predict_proba(eval_data.drop(columns=["id"]))[:, clf_xgb_optimized.classes_ == True].squeeze()


# Make the submission file
submission_df = pd.DataFrame({"id": eval_data["id"], "Label": y_preds})
submission_df["id"] = submission_df["id"].astype(int)
submission_df.to_csv("xgb_optimized.csv", sep=",", index=False)