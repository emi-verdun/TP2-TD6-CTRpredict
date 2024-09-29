import pandas as pd
import gc
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb

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

# Entrenamiento y evaluación del modelo XGBoost
xgb_params = {'colsample_bytree': 0.75,
              'gamma': 0.5,
              'learning_rate': 0.075,
              'max_depth': 8,
              'min_child_weight': 1,
              'n_estimators': 500,
              'reg_lambda': 0.5,
              'subsample': 0.75,
              }

clf_xgb = xgb.XGBClassifier(objective = 'binary:logistic',
                            seed = 1234,
                            eval_metric = 'auc',
                            **xgb_params)

clf_xgb.fit(X_train, y_train, verbose = 100, eval_set = [(X_train, y_train), (X_val, y_val)])

preds_val_xgb = clf_xgb.predict_proba(X_test_val)[:, clf_xgb.classes_ == True]
print("AUC test score - XGBoost:", roc_auc_score(y_test_val, preds_val_xgb)) 

# Predict on the evaluation set
eval_data = test_data.select_dtypes(include='number')
y_preds = clf_xgb.predict_proba(eval_data.drop(columns=["id"]))[:, clf_xgb.classes_ == True].squeeze()


# Make the submission file
submission_df = pd.DataFrame({"id": eval_data["id"], "Label": y_preds})
submission_df["id"] = submission_df["id"].astype(int)
submission_df.to_csv("xgb_todos_datasets.csv", sep=",", index=False)