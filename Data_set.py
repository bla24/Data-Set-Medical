# ==============================
# 1. IMPORTATION DES LIBRAIRIES
# ==============================

import pandas as pd  # Manipulation des données (DataFrame)
import numpy as np   # Calcul numérique

import matplotlib.pyplot as plt  # Visualisation graphique
import seaborn as sns  # Visualisation avancée

from sklearn.model_selection import train_test_split  # Division train/test
from sklearn.preprocessing import StandardScaler  # Normalisation
from sklearn.preprocessing import LabelEncoder  # Encodage cible si nécessaire

from sklearn.linear_model import LogisticRegression  # Modèle 1
from sklearn.ensemble import RandomForestClassifier  # Modèle 2
from sklearn.svm import SVC  # Modèle 3

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from sklearn.feature_selection import SelectKBest, f_classif  # Feature selection

# ==============================
# 2. CHARGEMENT DU DATASET
# ==============================

df = pd.read_excel("dataset_maladie_40vars.xlsx")  # Lecture du fichier Excel

print("Aperçu des données :")
print(df.head())  # Affiche les 5 premières lignes

# ==============================
# 3. ANALYSE EXPLORATOIRE (EDA)
# ==============================

# Histogrammes pour toutes les variables numériques
df.hist(figsize=(15, 12))  # Taille globale des graphiques
plt.suptitle("Distributions des variables numériques")  # Titre global
plt.show()

# Matrice de corrélation
plt.figure(figsize=(12,10))
corr_matrix = df.corr(numeric_only=True)  # Corrélation entre variables numériques
sns.heatmap(corr_matrix, cmap="coolwarm")  # Carte thermique
plt.title("Matrice de corrélation")
plt.show()

# ==============================
# 4. ANALYSE DE LA VARIABLE CIBLE
# ==============================

target_column = "PATHOLOGIE"  # Remplacer si le nom est différent

print("\nRépartition de la variable cible :")
print(df[target_column].value_counts())

sns.countplot(x=target_column, data=df)
plt.title("Distribution de la variable cible")
plt.show()

print("\nProportion (%) des classes :")
print(df[target_column].value_counts(normalize=True) * 100)

# ==============================
# 5. GESTION DES VALEURS MANQUANTES
# ==============================

print("\nValeurs manquantes par colonne :")
print(df.isnull().sum())

# Remplacement des valeurs manquantes numériques par la médiane
df.fillna(df.median(numeric_only=True), inplace=True)

# ==============================
# 6. GESTION DES OUTLIERS (IQR)
# ==============================

Q1 = df.quantile(0.25, numeric_only=True)  # 1er quartile
Q3 = df.quantile(0.75, numeric_only=True)  # 3e quartile
IQR = Q3 - Q1  # Intervalle interquartile

# Suppression des lignes contenant des outliers
df_clean = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

print("\nNombre de lignes après suppression des outliers :", df_clean.shape[0])

# ==============================
# 7. ENCODAGE DES VARIABLES CATEGORIELLES
# ==============================

# Transformation des variables catégorielles en variables numériques
df_encoded = pd.get_dummies(df_clean, drop_first=True)

# ==============================
# 8. SEPARATION FEATURES / TARGET
# ==============================

X = df_encoded.drop(target_column, axis=1)  # Variables explicatives
y = df_encoded[target_column]  # Variable cible

# ==============================
# 9. DIVISION TRAIN / TEST (80/20)
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,  # 20% test
    random_state=42,  # Reproductibilité
    stratify=y  # Conserve la proportion des classes
)

# ==============================
# 10. NORMALISATION DES DONNEES
# ==============================

scaler = StandardScaler()  # Création du scaler

X_train_scaled = scaler.fit_transform(X_train)  # Apprentissage sur train
X_test_scaled = scaler.transform(X_test)  # Transformation du test

# ==============================
# 11. ENTRAINEMENT DES MODELES
# ==============================

# ---- 11.1 Régression Logistique ----
log_model = LogisticRegression(max_iter=1000)  # max_iter augmenté pour convergence
log_model.fit(X_train_scaled, y_train)

# ---- 11.2 Random Forest ----
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# ---- 11.3 SVM ----
svm_model = SVC(probability=True)
svm_model.fit(X_train_scaled, y_train)

# ==============================
# 12. EVALUATION DES MODELES
# ==============================

def evaluate_model(model, X_test_data, model_name):
    """
    Fonction d'évaluation :
    - Affiche rapport classification
    - Affiche matrice de confusion
    - Calcule ROC-AUC
    """

    y_pred = model.predict(X_test_data)  # Prédictions classes
    y_prob = model.predict_proba(X_test_data)[:, 1]  # Probabilités classe 1

    print(f"\n===== {model_name} =====")
    print(classification_report(y_test, y_pred))  # Precision, Recall, F1

    print("Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_prob)
    print("ROC-AUC :", roc_auc)

# Evaluation des 3 modèles
evaluate_model(log_model, X_test_scaled, "Régression Logistique")
evaluate_model(rf_model, X_test, "Random Forest")
evaluate_model(svm_model, X_test_scaled, "SVM")

# ==============================
# 13. FEATURE SELECTION (BONUS)
# ==============================

selector = SelectKBest(score_func=f_classif, k=10)  # Sélection des 10 meilleures variables
X_new = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]
print("\nTop 10 variables sélectionnées :")
print(selected_features)

# ==============================
# 14. IMPORTANCE DES VARIABLES (INTERPRETABILITE)
# ==============================

importances = rf_model.feature_importances_  # Importance Random Forest

feature_importance_df = pd.Series(importances, index=X.columns)
feature_importance_df = feature_importance_df.sort_values(ascending=False)

print("\nImportance des variables (Random Forest) :")
print(feature_importance_df.head(10))

plt.figure(figsize=(10,6))
feature_importance_df.head(10).plot(kind='bar')
plt.title("Top 10 variables importantes")
plt.show()