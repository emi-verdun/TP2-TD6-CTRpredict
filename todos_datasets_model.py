import pandas as pd
import gc
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

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
train_data = pd.concat([ctr21, ctr20, ctr19, ctr18, ctr17, ctr16, ctr15], ignore_index=True)

# Load the test data
test_data = pd.read_csv("ctr_test.csv")

# Train a tree on the train data
train_data = train_data.sample(frac=1/7, random_state=2345)
y = train_data["Label"]
x = train_data.drop(columns=["Label"])
x = x.select_dtypes(include='number')
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=2345)
del train_data
gc.collect()

cls = make_pipeline(SimpleImputer(), DecisionTreeClassifier(max_depth=8, random_state=2345))
cls.fit(X_train, y_train)

pred_on_val = cls.predict_proba(X_val)[:,1]
auc_roc = roc_auc_score(y_val, pred_on_val)
print(auc_roc)

# Predict on the evaluation set
eval_data = test_data.select_dtypes(include='number')
y_preds = cls.predict_proba(eval_data.drop(columns=["id"]))[:, cls.classes_ == 1].squeeze()


# Make the submission file
submission_df = pd.DataFrame({"id": eval_data["id"], "Label": y_preds})
submission_df["id"] = submission_df["id"].astype(int)
submission_df.to_csv("todos_los_datasets_model.csv", sep=",", index=False)