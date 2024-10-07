import pandas as pd
import gc
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)


## Cargamos los datos

# Cargar datos para entrenamiento
ctr15 = pd.read_csv("ctr_15.csv")
ctr16 = pd.read_csv("ctr_16.csv")
ctr17 = pd.read_csv("ctr_17.csv")
ctr18 = pd.read_csv("ctr_18.csv")
ctr19 = pd.read_csv("ctr_19.csv")
ctr20 = pd.read_csv("ctr_20.csv")
ctr21 = pd.read_csv("ctr_21.csv")
train_data = pd.concat([ctr21, ctr20, ctr19, ctr18, ctr17, ctr16, ctr15], ignore_index=True)

# Cargar datos para test
test = pd.read_csv("ctr_test.csv")
test_data = test.drop(columns=["id"])

# Separo Label de los demás atributos
train_data = train_data.sample(frac=7/7, random_state=2345)
y = train_data["Label"]
x = train_data.drop(columns=["Label"])


## Preprocesamiento

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

# Procesar timestamp (auction_time) para test_data
test_data['auction_time'] = pd.to_datetime(test_data['auction_time'], unit='s')
test_data['hour'] = test_data['auction_time'].dt.hour
test_data['day_of_week'] = test_data['auction_time'].dt.dayofweek
test_data['is_weekend'] = test_data['auction_time'].dt.dayofweek >= 5
test_data = test_data.drop(columns=['auction_time'])

# Crear interacción de características en test_data
test_data['bidfloor_age_interaction'] = test_data['auction_bidfloor'] * test_data['auction_age']

# Utilizamos solo las columnas numéricas y categóricas
x = x.select_dtypes(include=['number', 'category'])
test_data = test_data.select_dtypes(include=['number', 'category'])


# Separación de un conjunto de validación
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=2345)


## Visualizaciones Análisis Exploratorio

#Figura 1
#Distribución de impresiones por hora del día
train_data['auction_time'] = pd.to_datetime(train_data['auction_time'], unit='s')
train_data['hour'] = train_data['auction_time'].dt.hour
sns.countplot(x='hour', data=train_data)
plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.2)
plt.title('Distribución de impresiones por hora del día')
plt.xlabel('Hora del día')
plt.ylabel('Cantidad de impresiones')
plt.show()

#Figura 2
#Cantidad de clics (Label=1) por hora del día
train_data_label_1 = train_data[train_data['Label'] == 1]


sns.countplot(x='hour', data=train_data_label_1, color="orange")
plt.title('Cantidad de clics (Label=1) por hora del día')
plt.xlabel('Hora del día')
plt.ylabel('Cantidad de clics')
plt.show()


#Eliminamos el conjunto de train sin separar
del train_data
gc.collect()


## Preprocesamiento (luego de separar conjunto de validación y entrenamiento)

# Identificar columnas numéricas
numeric_columns = X_train.select_dtypes(include=['float64', 'int32']).columns

# Crear imputers para variables categóricas (no imputamos numéricas porque el modelo no mostraba mejoras significativas)
imputer_cat = SimpleImputer(strategy='most_frequent')

# Imputar valores faltantes para el conjunto de entrenamiento
for col in categorical_columns:
    if X_train[col].isnull().sum() > 0:
        X_train[col] = X_train[col].cat.add_categories('Missing')  # Agregar 'Missing' como categoría
        X_train[col] = X_train[col].fillna('Missing')  # Rellenar con 'Missing'

# Imputar valores faltantes para el conjunto de validación
for col in categorical_columns:
    if X_val[col].isnull().sum() > 0:
        X_val[col] = X_val[col].cat.add_categories('Missing')  # Agregar 'Missing' como categoría
        X_val[col] = X_val[col].fillna('Missing')  # Rellenar con 'Missing'

# Imputar valores faltantes para el conjunto de prueba
for col in categorical_columns:
    if test_data[col].isnull().sum() > 0:
        test_data[col] = test_data[col].cat.add_categories('Missing')  # Agregar 'Missing' como categoría
        test_data[col] = test_data[col].fillna('Missing')  # Rellenar con 'Missing'


#Trasformación de creative_height en los 3 conjuntos de datos
if not(X_train['creative_height'].empty):
    X_train['creative_height_squared'] = X_train['creative_height'] ** 2

if not(X_val['creative_height'].empty):
    X_val['creative_height_squared'] = X_val['creative_height'] ** 2

if not(test_data['creative_height'].empty):
    test_data['creative_height_squared'] = test_data['creative_height'] ** 2


## Visualizaciones Análisis Exploratorio

#Figura 3
#Correlación de Variables con y_train

data_with_labels = pd.concat([X_train, y_train], axis=1)

# Calcular la correlación de Pearson entre las características e y_train
numerical_data = data_with_labels.select_dtypes(include=['float64', 'int64'])
correlations = numerical_data.corr()['Label'].drop('Label')  # Excluir la autocorrelación de 'Label'

# Ordenar las correlaciones de mayor a menor
correlations_sorted = correlations.abs().sort_values(ascending=False)

# Graficar las correlaciones
plt.figure(figsize=(10, 8))
correlations_sorted.plot(kind='bar')
plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.2)
plt.title('Correlación de Variables con y_train')
plt.xlabel('Variables')
plt.ylabel('Coeficiente de Correlación de Pearson')
plt.show()


## Entrenar modelo
cls = lgb.LGBMClassifier(n_estimators=200, 
                                       max_depth=14, 
                                       learning_rate=0.01, 
                                       boosting_type='dart', 
                                       random_state=2345,
                                       is_unbalance=True)
cls.fit(X_train, y_train)

#Predecimos sobre validación y vemos auc_roc
pred_on_val = cls.predict_proba(X_val)[:,1]
auc_roc = roc_auc_score(y_val, pred_on_val)
print(auc_roc)

## Atributos más significativos
#Figura 4
importancia = cls.feature_importances_

# Armamos un dataframe con las variables y su importacia
importancia_df = pd.DataFrame({
    'Variables': X_train.columns,
    'Importancia': importancia
}).sort_values(by='Importancia', ascending=False) #Ordenamos la importancia de mayor a menor

# Nos quedamos con las 10 variables más importantes
top_10 = importancia_df.head(10)

# Graficamos
plt.figure(figsize=(10, 6))
plt.barh(top_10['Variables'], top_10['Importancia'], color='skyblue')
plt.xlabel('Importancia')
plt.ylabel('Variables')
plt.title('Top 10 Variables más Importantes')
plt.gca().invert_yaxis() 
plt.show()


# Predict on the evaluation set
y_preds = cls.predict_proba(test_data)[:, 1] 


# Make the submission file
submission_df = pd.DataFrame({"id": test["id"], "Label": y_preds})
submission_df["id"] = submission_df["id"].astype(int)
submission_df.to_csv("lightGBM_8.csv", sep=",", index=False)