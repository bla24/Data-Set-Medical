# import des librairies pour notre projet

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Importer Random Forest
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import random

# Déclaration des variables
# base contenant les variables suivantes
# -	Variables de 1 à 40
# -	PATHOLOGIE : 0 : Absence / 1 : Présence

noms_variables = ["V1","V2","V3","V4","sexe","degre_rougeur_nez"]

# Chargement du dataset binaire
# Chargement du dataset binaire
donnees_data_frame = pd.read_csv('dataset_maladie_40vars.xlsx', delimiter="\t")
print(donnees_data_frame)
data = load_breast_cancer()
X, y = data.data, data.target

# Diviser le dataset en ensembles d'entraînement et de test:
# 80% de train set et 20% de test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)