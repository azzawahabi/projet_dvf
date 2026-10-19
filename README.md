# Prédiction du prix au m² des logements dans l'Ain (DVF 2022-2025)

Ce projet de machine learning estime le prix au m² d'une maison ou d'un appartement dans le département de l'Ain (01). Il s'appuie sur les ventes immobilières réelles publiées par l'État (base **DVF**, Demandes de valeurs foncières).

Le modèle final atteint une **erreur médiane de 15,1 %** sur les ventes de 2025, qu'il n'a jamais vues. La méthode de référence (prix médian de la commune) fait 17,9 %. Il fournit aussi une **fourchette de prix à 80 %**, qui contient le vrai prix dans 82 % des cas.

---

## Résultats

| Étape | Modèle | Erreur médiane 2025 | Ventes à ±20 % |
|---|---|---|---|
| Référence | Prix médian commune × type | 17,9 % | 55,4 % |
| 2 | LightGBM (bien + commune) | 16,3 % | 59,6 % |
| 3 | + données externes (revenus, distances) | 16,1 % | 59,9 % |
| 4 | + tendance temporelle par zone | 16,6 % | 59,0 % |
| **5** | **Perte L1 + prix des voisins (modèle final)** | **15,1 %** | **62,0 %** |

*Erreur médiane : la moitié des estimations ont une erreur plus petite. Test : 7 648 ventes de 2025.*

---

## Structure du projet

```
projet_dvf/
├── dvf_etape1_vscode.ipynb                  # Chargement et nettoyage
├── dvf_etape2_modeles.ipynb                 # Comparaison des modèles
├── dvf_etape3_donnees_externes.ipynb        # Données externes, ablation
├── dvf_etape4_derive_temporelle.ipynb       # Correction de l'effet du temps
├── dvf_etape5_voisins_perte_robuste.ipynb   # Modèle final, SHAP, predire()
├── requirements.txt
└── data/
    ├── ValeursFoncieres-2022.txt ... 2025.txt   # à télécharger (voir plus bas)
    └── externe/
        ├── communes_ain.csv                      # coordonnées, population, distances
        ├── valeurs_trimestrielles.csv            # indice Notaires-Insee
        ├── caractéristiques.csv                  # libellés de l'indice
        ├── DS_FILOSOFI_CC_data.csv               # revenus Insee Filosofi 2021
        └── DS_FILOSOFI_CC_metadata.csv
```

Les fichiers produits par les notebooks (`dvf_clean.parquet`, `etapeX_*.csv`, etc.) sont écrits dans `data/`.

---

## Installation

```bash
git clone https://github.com/<ton-compte>/projet_dvf.git
cd projet_dvf
pip install -r requirements.txt
```

Python 3.10 ou plus récent.

## Données

Les fichiers DVF bruts (environ 500 Mo chacun) ne sont pas inclus dans le dépôt.

1. **DVF** : télécharger les fichiers `ValeursFoncieres-2022.txt` à `2025.txt` sur [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/) et les placer dans `data/`.
2. **Indice Notaires-Insee** des prix des logements anciens (Auvergne-Rhône-Alpes, trimestriel) : [insee.fr](https://www.insee.fr/fr/statistiques/series/105071770).
3. **Filosofi 2021** (revenus des ménages par commune) : [insee.fr](https://www.insee.fr/fr/statistiques/7756729).
4. **Coordonnées et population des communes** : récupérées automatiquement depuis l'API [geo.api.gouv.fr](https://geo.api.gouv.fr).

## Utilisation

Lancer les notebooks **dans l'ordre**, de l'étape 1 à l'étape 5. Chaque étape lit les fichiers produits par la précédente.

Exemple avec le modèle final (étape 5) :

```python
predire('GEX', 'Maison', surface=120, pieces=5, terrain=600, nb_dependances=1)
# {'prix_m2': 5418, 'prix_total': 650000, 'intervalle_80%': (510000, 830000)}
```

---

## Méthode

### 1. Nettoyage des données
Dans DVF, une vente occupe plusieurs lignes (une par parcelle, par local, par type de terrain), et le prix est répété sur chacune. Le nettoyage reconstruit les ventes puis les filtre :

| Filtre | Ventes restantes |
|---|---|
| Ventes dans les fichiers bruts (178 618 lignes) | 60 070 |
| Ventes classiques uniquement (sans échange, VEFA, enchère, expropriation) | 55 149 |
| Sans local commercial (prix impossible à séparer) | 51 864 |
| Un seul logement (maison ou appartement) | 31 026 |
| Règles de cohérence (prix, surface, pièces, prix au m²) | 30 742 |
| Sans le 1 % le moins cher et le 1 % le plus cher, par type et par année | **30 125** |

### 2. Protocole d'évaluation (sans fuite de données)
- Découpage **temporel** : apprentissage sur 2022-2024, test sur 2025.
- Les choix (variables, réglages) se font sur un backtest 2024 ; 2025 n'est évalué qu'une fois.
- Les données externes sont décalées de leur délai de publication. Exemple : l'indice Notaires-Insee est décalé de 2 trimestres.
- Gains testés par un intervalle de confiance bootstrap apparié à 95 %.

### 3. Modèles
Baselines (médiane par type, médiane commune × type), Ridge, CatBoost, LightGBM.

### 4. Principaux enseignements
- **Les revenus des communes** (Insee Filosofi) améliorent un peu le modèle.
- **L'indice régional des prix le dégrade** : il suit toute la région, alors que l'Ain a sa propre dynamique (hausse dans le Pays de Gex, stabilité ailleurs).
- **Aucune correction de tendance** (pondération, extrapolation, réentraînement trimestriel) ne bat le modèle simple sur 2025.
- **La perte L1** (viser la médiane plutôt que la moyenne) apporte le gain principal : −1 point d'erreur, et le biais passe de −4,2 % à −0,6 %. Les ventes atypiques (biens à rénover, ventes familiales) tiraient le modèle vers le bas.
- **Intervalles de prédiction conformes** (méthode Mondrian, par type de bien × zone) : 82 % de couverture pour un objectif de 80 %.

---

## Limites

- DVF ne décrit pas **l'état du bien** : les biens à rénover restent mal prédits.
- Le **Pays de Gex** reste sous-estimé (biais de −5,9 % en 2025).
- Seulement 4 années de données ; revenus Insee manquants pour 31,5 % des ventes.

## Pistes d'amélioration

- Ajouter le **DPE** de l'ADEME (classe énergie, année de construction).
- Utiliser le **DVF géolocalisé** pour mesurer le prix des ventes les plus proches.
- Construire un indice de prix propre au Pays de Gex.

---

## Outils

Python · pandas · NumPy · scikit-learn · LightGBM · CatBoost · SHAP · Matplotlib · Jupyter

## Sources des données

- DVF : Direction générale des Finances publiques, [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/), Licence Ouverte
- Insee : indices Notaires-Insee, Filosofi 2021
- API Découpage administratif : [geo.api.gouv.fr](https://geo.api.gouv.fr)

## Auteur

Wahabi Azza · ESSAI · 2026
