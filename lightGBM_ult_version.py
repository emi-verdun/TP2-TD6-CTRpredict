import pandas as pd
import gc
from sklearn.impute import SimpleImputer
import lightgbm as lgb
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
test = pd.read_csv("ctr_test.csv")
test_data = test.drop(columns=["id"])

# Train a tree on the train data
train_data = train_data.sample(frac=7/7, random_state=2345)
y = train_data["Label"]
x = train_data.drop(columns=["Label"])


# Convertir variables categóricas al tipo adecuado
categorical_columns = [col for col in x.columns if 'categorical' in col]
for col in categorical_columns:
    x[col] = x[col].astype('category')

# Procesar timestamp (auction_time)
x['auction_time'] = pd.to_datetime(x['auction_time'], unit='s')
x['hour'] = x['auction_time'].dt.hour
x['day_of_week'] = x['auction_time'].dt.dayofweek
x['is_weekend'] = x['auction_time'].dt.dayofweek >= 5
x = x.drop(columns=['auction_time'])

# Crear interacción de características
x['bidfloor_age_interaction'] = x['auction_bidfloor'] * x['auction_age']

# Convertir variables categóricas al tipo adecuado para test_data
categorical_columns_test = [col for col in test_data.columns if 'categorical' in col]
for col in categorical_columns_test:
    test_data[col] = test_data[col].astype('category')

# Procesar timestamp (auction_time)
test_data['auction_time'] = pd.to_datetime(test_data['auction_time'], unit='s')
test_data['hour'] = test_data['auction_time'].dt.hour
test_data['day_of_week'] = test_data['auction_time'].dt.dayofweek
test_data['is_weekend'] = test_data['auction_time'].dt.dayofweek >= 5
test_data = test_data.drop(columns=['auction_time'])

# Crear interacción de características en test_data
test_data['bidfloor_age_interaction'] = test_data['auction_bidfloor'] * test_data['auction_age']


# Eliminar variables no numéricas excepto las categóricas
x = x.select_dtypes(include=['number', 'category'])
test_data = test_data.select_dtypes(include=['number', 'category'])



X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=2345)
del train_data
gc.collect()



# Identificar columnas numéricas
numeric_columns = X_train.select_dtypes(include=['float64', 'int32']).columns

# Crear imputers separados para variables numéricas y categóricas
#imputer_num = SimpleImputer(strategy='median')
imputer_cat = SimpleImputer(strategy='most_frequent')

# Imputar valores faltantes para el conjunto de entrenamiento
#X_train[numeric_columns] = imputer_num.fit_transform(X_train[numeric_columns])
for col in categorical_columns:
    if X_train[col].isnull().sum() > 0:
        X_train[col] = X_train[col].cat.add_categories('Missing')  # Agregar 'Missing' como categoría
        X_train[col] = X_train[col].fillna('Missing')  # Rellenar con 'Missing'

# Imputar valores faltantes para el conjunto de validación
#X_val[numeric_columns] = imputer_num.transform(X_val[numeric_columns])  # Usar el mismo imputador
for col in categorical_columns:
    if X_val[col].isnull().sum() > 0:
        X_val[col] = X_val[col].cat.add_categories('Missing')  # Asegurarse de que la categoría 'Missing' exista
        X_val[col] = X_val[col].fillna('Missing')  # Rellenar con 'Missing'

#numeric_columns = test_data.select_dtypes(include=['float64', 'int32']).columns
# Imputar valores faltantes para el conjunto de prueba
#test_data[numeric_columns] = imputer_num.transform(test_data[numeric_columns])  # Usar el mismo imputador
for col in categorical_columns:
    if test_data[col].isnull().sum() > 0:
        test_data[col] = test_data[col].cat.add_categories('Missing')  # Asegurarse de que la categoría 'Missing' exista
        test_data[col] = test_data[col].fillna('Missing')  # Rellenar con 'Missing'


if not(X_train['creative_height'].empty):
    X_train['creative_height_squared'] = X_train['creative_height'] ** 2

if not(X_val['creative_height'].empty):
    X_val['creative_height_squared'] = X_val['creative_height'] ** 2

if not(test_data['creative_height'].empty):
    test_data['creative_height_squared'] = test_data['creative_height'] ** 2


cls = make_pipeline(lgb.LGBMClassifier(n_estimators=200, 
                                       max_depth=14, 
                                       learning_rate=0.01, 
                                       boosting_type='dart', 
                                       random_state=2345,
                                       is_unbalance=True))
cls.fit(X_train, y_train)


pred_on_val = cls.predict_proba(X_val)[:,1]
auc_roc = roc_auc_score(y_val, pred_on_val)
print(auc_roc)


# Predict on the evaluation set
y_preds = cls.predict_proba(test_data)[:, 1] 


# Make the submission file
submission_df = pd.DataFrame({"id": test["id"], "Label": y_preds})
submission_df["id"] = submission_df["id"].astype(int)
submission_df.to_csv("lightGBM_8.csv", sep=",", index=False)