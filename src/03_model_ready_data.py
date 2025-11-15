import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import joblib

# Caminho absoluto para o dataset e para salvar o modelo
dataset_path = os.path.join("data", "rh_turnover_dataset.csv")
model_path = os.path.join(os.path.dirname(__file__), "model_rf.pkl")

# Verifica se o dataset existe
if not os.path.exists(dataset_path):
    raise FileNotFoundError("⚠️ O arquivo 'data/rh_turnover_dataset.csv' não foi encontrado. Rode o script '01_data_generation.py' primeiro.")

# Carrega os dados
print("🔍 Carregando dados...")
df = pd.read_csv(dataset_path)
print("✅ Dados carregados:", df.shape)

# Separação de features e target
X = df.drop(columns=["turnover"])
y = df["turnover"]

# Identificação de colunas numéricas e categóricas
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Pré-processamento
preprocess = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
    remainder="passthrough"
)

# Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Pipeline com RandomForest
print("🧠 Treinando modelo...")
rf = Pipeline([
    ("prep", preprocess),
    ("clf", RandomForestClassifier(n_estimators=300, random_state=42))
])
rf.fit(X_train, y_train)
print("✅ Modelo treinado.")

# Avaliação
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

print("📊 Avaliação do modelo:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# Salvamento do modelo
joblib.dump(rf, model_path)
print(f"💾 Modelo salvo em {model_path}")
