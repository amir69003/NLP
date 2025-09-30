# 🧠 Classifieur de textes avec RNN (PyTorch)

## 📌 Contexte académique
Projet réalisé dans le cadre du **Master 2 Intelligence Artificielle**  
Université Claude Bernard Lyon 1 – Département Informatique  
UE : **Intelligence Bio-Inspirée**

## 🎯 Objectif
Développer un **classifieur de textes** basé sur un **Réseau de Neurones Récurrent (RNN)** en utilisant **PyTorch**.  
L’objectif principal était de traiter un dataset de phrases étiquetées par sentiments et de concevoir un modèle capable de **prédire correctement l’émotion associée à un texte**.

## ✨ Fonctionnalités implémentées
- 🔹 **Chargement et prétraitement des données**
  - Découpage du dataset en `train`, `validation`, et `test`
  - Fonction `load_file(file)` pour parser les textes et labels
  - Nettoyage des mots rares (filtrage TF-IDF)
- 🔹 **Encodage des données**
  - Création d’un dictionnaire `mot → id`
  - Gestion du **padding** pour uniformiser les phrases
  - Encodage **one-hot** et **embedding learnable**
- 🔹 **Modèle RNN (PyTorch)**
  - Implémentation d’une classe `RNN(nn.Module)`
  - Ajout d’une couche d’embedding pour améliorer la représentation des mots
  - Support du traitement par batch avec `DataLoader`
- 🔹 **Boucle d’apprentissage**
  - Entraînement par `epoch/batch` avec suivi des courbes
  - Validation croisée pour prévenir l’overfitting
  - Utilisation de fonctions de perte adaptées pour gérer le déséquilibre des classes
- 🔹 **Évaluations et visualisations**
  - Courbes **accuracy & loss** (train/validation)
  - **Matrice de confusion** sur le jeu de test
  - Visualisation des embeddings (PCA, t-SNE)
- 🔹 **Approfondissements**
  - Représentation distribuée des mots
  - Expérimentation avec apprentissage auto-supervisé (prédiction `contexte → mot`)

## 🛠️ Stack technique
- **Langage :** Python 3  
- **Bibliothèques principales :** PyTorch, Torchtext, Scikit-learn, Matplotlib, NumPy  
- **Outils :** Jupyter Notebook, Git  
