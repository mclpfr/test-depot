# Projet de Prédiction des Accidents de la Route - MLOps

## Aperçu du Projet
Ce projet vise à prédire la gravité des accidents de la route en France en utilisant des techniques de machine learning. L'objectif est de fournir une estimation des urgences en temps réel pour les services de police et médicaux.

## Prérequis
- Python 3.10
- Bibliothèques Python : 
  - pandas
  - scikit-learn
  - numpy
  - requests
  - beautifulsoup4

## Installation
1. Clonez le dépôt
```bash
git clone https://github.com/mclpfr/mlops-road-accidents.git
```
2. Créez un environnement virtuel
```bash
bash scritps/setup_env.sh
```
3. Installez les dépendances
```bash
pip install -r requirements.txt
```

## Utilisation
### Extraction des Données
```bash
python src/extract_data.py
```

### Prétraitement des Données
```bash
python src/data_preprocessing.py
```

### Entraînement du Modèle
```bash
python src/model_training.py
```

# Description des Données d'Accidents

## Vue d'ensemble
Le fichier fusionné `data/raw/accidents_<année>.csv` contient des colonnes provenant de plusieurs fichiers sources : caract, lieux, usagers et vehicules.

## Notes importantes
- La colonne `grav` (gravité) a été initialement codée de 1 à 4 :
  - 1 : Indemne
  - 2 : Blessé léger
  - 3 : Blessé grave
  - 4 : Tué
- Elle a été binarisée dans `prepare_data.py` en :
  - 0 : "Grave" (valeurs 3 et 4)
  - 1 : "Pas grave" (valeurs 1 et 2)

## Méthode de Fusion
Les fichiers sont fusionnés via la colonne `Num_Acc` (numéro d'accident unique) comme clé de jointure.

## Description détaillée des fichiers

### Fichier Caractéristiques (caract)

| Colonne | Description | Type | Exemple |
|---------|-------------|------|---------|
| Num_Acc | Numéro d'identifiant unique de l'accident | Chaîne | "202300000001" |
| jour | Jour de l'accident | Entier | 15 |
| mois | Mois de l'accident | Entier | 6 |
| an | Année de l'accident | Entier | 2023 |
| hrmn | Heure et minute de l'accident (format HHMM) | Chaîne | "1430" (14h30) |
| lum | Conditions d'éclairage (1 : Plein jour, 2 : Crépuscule, etc.) | Entier | 1 |
| dep | Département (code INSEE) | Chaîne | "75" (Paris) |
| com | Commune (code INSEE) | Chaîne | "75056" |
| agg | Localisation (1 : Hors agglomération, 2 : En agglomération) | Entier | 2 |
| int | Type d'intersection (1 : Hors intersection, 2 : Intersection en X, etc.) | Entier | 1 |
| atm | Conditions atmosphériques (1 : Normal, 2 : Pluie légère, etc.) | Entier | 1 |
| col | Type de collision (1 : Frontale, 2 : Par l'arrière, etc.) | Entier | 3 |
| lat | Latitude (coordonnées géographiques) | Flottant | 48.8566 |
| long | Longitude (coordonnées géographiques) | Flottant | 2.3522 |

### Fichier Lieux (lieux)

| Colonne | Description | Type | Exemple |
|---------|-------------|------|---------|
| Num_Acc | Numéro d'identifiant de l'accident (clé de fusion) | Chaîne | "202300000001" |
| catr | Catégorie de route (1 : Autoroute, 2 : Route nationale, etc.) | Entier | 3 |
| voie | Numéro de la route | Chaîne | "N7" |
| v1 | Indice numérique de la route | Entier | 0 |
| v2 | Indice alphanumérique de la route | Chaîne | "A" |
| circ | Régime de circulation (1 : Sens unique, 2 : Bidirectionnel, etc.) | Entier | 2 |
| nbv | Nombre total de voies de circulation | Entier | 2 |
| vosp | Présence d'une voie réservée (0 : Sans, 1 : Piste cyclable, etc.) | Entier | 0 |
| prof | Profil en long (1 : Plat, 2 : Pente, etc.) | Entier | 1 |
| plan | Tracé en plan (1 : Droit, 2 : Courbe à gauche, etc.) | Entier | 1 |
| lartpc | Largeur du terre-plein central (en mètres) | Entier | 0 |
| larrout | Largeur de la chaussée (en mètres) | Entier | 7 |
| surf | État de la surface (1 : Normale, 2 : Mouillée, etc.) | Entier | 1 |
| infra | Infrastructure (0 : Aucune, 1 : Souterrain, 2 : Pont, etc.) | Entier | 0 |
| situ | Situation de l'accident (1 : Sur chaussée, 2 : Sur accotement, etc.) | Entier | 1 |

### Fichier Usagers (usagers)

| Colonne | Description | Type | Exemple |
|---------|-------------|------|---------|
| Num_Acc | Numéro d'identifiant de l'accident (clé de fusion) | Chaîne | "202300000001" |
| id_vehicule | Identifiant unique du véhicule | Chaîne | "1" |
| num_veh | Numéro du véhicule | Chaîne | "A01" |
| place | Place occupée dans le véhicule (1 : Conducteur, 2 : Passager avant, etc.) | Entier | 1 |
| catu | Catégorie d'usager (1 : Conducteur, 2 : Passager, 3 : Piéton) | Entier | 1 |
| grav | Gravité de la blessure (0 : grave, 1 : pas grave) | Entier | 1 |
| sexe | Sexe de l'usager (1 : Masculin, 2 : Féminin) | Entier | 1 |
| an_nais | Année de naissance de l'usager | Entier | 1985 |
| trajet | Motif du déplacement (1 : Domicile-travail, 2 : Domicile-école, etc.) | Entier | 5 |
| secu1 | Premier équipement de sécurité (1 : Ceinture, 2 : Casque, etc.) | Entier | 1 |
| secu2 | Deuxième équipement de sécurité (si applicable) | Entier | 0 |
| secu3 | Troisième équipement de sécurité (si applicable) | Entier | 0 |
| locp | Localisation du piéton (0 : Non applicable, 1 : Sur chaussée, etc.) | Entier | 0 |
| actp | Action du piéton (0 : Non applicable, 1 : Traversant, etc.) | Chaîne | "0" |
| etatp | État du piéton (0 : Non applicable, 1 : Seul, 2 : Accompagné, etc.) | Entier | 0 |

### Fichier Véhicules (vehicules)

| Colonne | Description | Type | Exemple |
|---------|-------------|------|---------|
| Num_Acc | Numéro d'identifiant de l'accident (clé de fusion) | Chaîne | "202300000001" |
| id_vehicule | Identifiant unique du véhicule | Chaîne | "1" |
| num_veh | Numéro du véhicule | Chaîne | "A01" |
| senc | Sens de circulation (1 : Sens 1, 2 : Sens 2, etc.) | Entier | 1 |
| catv | Catégorie du véhicule (1 : Bicyclette, 7 : VL seul, 33 : Moto >125cm3, etc.) | Entier | 7 |
| obs | Obstacle fixe heurté (0 : Aucun, 1 : Véhicule en stationnement, etc.) | Entier | 0 |
| obsm | Obstacle mobile heurté (0 : Aucun, 1 : Piéton, 2 : Véhicule, etc.) | Entier | 0 |
| choc | Point de choc initial (0 : Aucun, 1 : Avant, 2 : Arrière, etc.) | Entier | 1 |
| manv | Manœuvre avant l'accident (1 : Sans changement de direction, 2 : Dépassement, etc.) | Entier | 1 |
| motor | Type de motorisation (1 : Essence, 2 : Diesel, etc.) | Entier | 2 |
| occutc | Nombre d'occupants dans un transport en commun | Entier | 0 |
