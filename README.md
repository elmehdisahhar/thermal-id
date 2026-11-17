# Identification thermique d'un bâtiment par apprentissage profond

Assessment de stage coEnergy : prédire, à partir de 8 heures de mesures, les indicateurs thermiques
intrinsèques d'un bâtiment.

- **Entrées** (480 pas d'une minute) : température intérieure $T_{int}(t)$, arrondie à 0,01 °C comme dans
  l'exemple de chargement de l'énoncé, et puissance de chauffage $U(t)$.
- **Sorties** : HLC et $H_b = (\text{HLC}_{mit}, \text{HLC}_{ext})$ en W/K ; constantes de temps $\tau_b$, $\tau_n$,
  $\tau_\infty$ en heures.
- **Données** : 1 000 000 de simulations `quentin_house`, en 20 batches de 50 000.
- **Contrainte physique** : $\text{HLC} = \text{HLC}_{mit} + \text{HLC}_{ext}$, respectée exactement : le modèle
  prédit $H_b$ et le HLC est leur somme.

## Résultat en bref

Test final : 100 000 simulations jamais vues (batches 19–20), évaluées une seule fois. MAPE en %, plus bas = mieux.

| Modèle | HLC | HLC_mit | HLC_ext | τ_b | τ_n | τ_∞ | R² HLC |
|---|---|---|---|---|---|---|---|
| Moyenne du train | 11,13 | 19,02 | 12,81 | 15,98 | 14,22 | 18,99 | 0,00 |
| Régression linéaire (degré 2) | 8,57 | 16,07 | 10,80 | 15,37 | 12,59 | 18,42 | 0,39 |
| MLP sur descripteurs | 8,37 | 16,00 | 10,00 | 15,09 | **12,31** | 18,17 | 0,41 |
| **CNN 1D (modèle retenu)** | **8,36** | **15,83** | **9,35** | **14,94** | 12,32 | **18,03** | **0,42** |

- Le CNN est le meilleur modèle sur cinq cibles sur six, avec un avantage net sur HLC$_{ext}$ (R² 0,44 contre
  0,36 pour le MLP).
- Les constantes de temps restent peu prédictibles (R² ≤ 0,25). Elles valent 11 à 49 h en médiane, bien plus que
  la fenêtre de 8 h : c'est une limite physique de la mesure, analysée dans le notebook 4.
- Les scores de validation et de test sont identiques : le choix du modèle n'a pas « collé » à la validation.

---

## 1. Organisation du dépôt

```
assesment_co_energy/
├── README.md                  ce fichier : installation, commandes, résultats
├── pyproject.toml, uv.lock    dépendances (uv), Python 3.12
├── notebooks/                 le rapport exécutable, à lire dans l'ordre
│   ├── 01_eda.ipynb           compréhension physique et exploration des données
│   ├── 02_baseline.ipynb      baseline : régression linéaire sur descripteurs physiques
│   ├── 03_cnn.ipynb           construction et vérification du CNN 1D, brique par brique
│   └── 04_resultats.ipynb     résultats sur le test, analyse des erreurs, limites, conclusion
├── src/thermal_id/            tout le code : modules typés et documentés, importés par les notebooks
│   ├── paths.py               chemins du projet (data/, runs/, results/)
│   ├── simul_io.py            lecture des batches bruts (reprise de util_assessement.py)
│   ├── data.py                format compact, arrondi de T, contrôles, normalisation, Dataset PyTorch
│   ├── prepare.py             étape 1 : préparation des 20 batches
│   ├── features.py            36 descripteurs physiques (forme de la réponse, identification modale)
│   ├── metrics.py             MAE, MAPE, R² par cible et par famille d'excitation
│   ├── baseline.py            étape 2 : régression linéaire (degrés 1 et 2)
│   ├── mlp_features.py        étape 2 bis : MLP sur les descripteurs
│   ├── model.py               CNN 1D résiduel (ThermalCNN)
│   ├── train.py               étape 3 : entraînement (CPU ou GPU, reprise possible)
│   ├── evaluate.py            étape 4 : évaluation d'un modèle sur la validation
│   ├── final.py               étape 5 : évaluation finale de tous les modèles sur le test
│   ├── linearity.py           analyse : linéarité, bruit, identification directe
│   ├── identifiability.py     analyse : courbes quasi identiques, cibles différentes ?
│   └── plots.py               graphiques du rapport
├── tests/                     18 tests unitaires et de bout en bout (pytest), sur données synthétiques
├── data/
│   ├── raw/quentin_house/     batches bruts fournis (batch_1.pt … batch_20.pt), non modifiés
│   ├── processed/             batches compacts, T à pleine précision (comparaison)
│   └── processed_t2/          batches compacts, T arrondi à 0,01 °C (modèle final)
├── runs/                      entraînements : best.pt, history.json, train_log.txt, config.json
│   ├── cnn_t2_400k/           modèle retenu (T arrondi)
│   └── cnn_cpu_400k/          même modèle sur T à pleine précision (comparaison)
├── results/                   métriques (JSON), prédictions, figures
│   └── t2/                    résultats sur T arrondi, dont final_test.json (test final)
└── docs/                      énoncé, article de référence, utilitaire fourni (util_assessement.py)
```

**Principe.** Tout le calcul est dans `src/thermal_id/`. Les notebooks importent ces modules et présentent la
démarche, les résultats et leur interprétation. Les étapes longues (préparation, entraînement) se lancent en
ligne de commande ; les notebooks relisent leurs sorties.

---

## 2. Installation

Prérequis : [uv](https://docs.astral.sh/uv/). Il installe lui-même Python 3.12 si nécessaire.

```powershell
uv sync                     # crée .venv et installe exactement les versions de uv.lock
uv run pytest -q            # 18 tests : ≈ 3 min à la première exécution, ≈ 15 s ensuite
```

Si `uv` n'est pas dans le PATH, remplacer `uv` par `python -m uv` dans toutes les commandes.

**Données.** Copier les 20 fichiers `batch_*.pt` du dossier `quentin_house` dans `data/raw/quentin_house/`.

---

## 3. Reproduire les résultats, étape par étape

Toutes les commandes se lancent depuis la racine du projet ; chaque étape lit les sorties de la précédente.
Durées mesurées sur un ordinateur portable (Intel i7-1355U, CPU seul).

| # | Étape | Commande | Durée | Produit |
|---|---|---|---|---|
| 1 | Préparation, T arrondi à 0,01 °C | `uv run python -m thermal_id.prepare --round-t 2 --out-dir data/processed_t2` | ≈ 1 min | batches compacts, `normalizer.json`, `manifest.json` |
| 2 | Baseline linéaire | `uv run python -m thermal_id.baseline --processed-dir data/processed_t2 --out results/t2` | ≈ 2 min | `baseline_val.json`, `baseline_deg{1,2}.pt` |
| 2 bis | MLP sur descripteurs | `uv run python -m thermal_id.mlp_features --processed-dir data/processed_t2 --out results/t2/mlp_features_val.json` | ≈ 2 min | métriques et poids du MLP |
| 3 | Entraînement du CNN | `uv run python -m thermal_id.train --processed-dir data/processed_t2 --n-train 400000 --epochs 10 --log-every 300 --out-dir runs/cnn_t2_400k` | ≈ 1 à 2 h | `runs/cnn_t2_400k/best.pt`, historique, journal |
| 4 | Évaluation sur la validation | `uv run python -m thermal_id.evaluate runs/cnn_t2_400k/best.pt --name cnn_t2_400k --processed-dir data/processed_t2` | ≈ 1 min | `results/cnn_t2_400k_val.json` |
| 5 | **Test final, une seule fois** | `uv run python -m thermal_id.final --processed-dir data/processed_t2 --results-dir results/t2 --cnn runs/cnn_t2_400k/best.pt` | ≈ 1 min | `results/t2/final_test.json`, `final_test_pred.pt` |

L'étape 5 évalue les quatre modèles sur le test, après les avoir figés. Elle ne doit être lancée qu'une fois,
avec le modèle choisi sur la validation.

### Comparaison avec T à pleine précision (notebook 4, section 5)

```powershell
uv run python -m thermal_id.prepare                                   # data/processed : T non arrondi
uv run python -m thermal_id.train --n-train 400000 --epochs 10 --out-dir runs/cnn_cpu_400k
uv run python -m thermal_id.evaluate runs/cnn_cpu_400k/best.pt --name cnn_400k
uv run python -m thermal_id.evaluate runs/cnn_cpu_400k/best.pt --name cnn_400k_round2 --round-t 2
```

### Analyses physiques (notebook 4, section 6)

| Analyse | Commande | Question posée |
|---|---|---|
| Linéarité et bruit | `uv run python -m thermal_id.linearity` (≈ 1 min ; `--modes 4` ajoute ≈ 13 min) | Le simulateur est-il linéaire et sans bruit ? Peut-on identifier le HLC directement ? |
| Identifiabilité | `uv run python -m thermal_id.identifiability` | Deux courbes presque identiques ont-elles des cibles proches ? |

### Options utiles de l'entraînement

```powershell
uv run python -m thermal_id.train --help
# reprise après interruption (mêmes arguments, plus --resume) :
uv run python -m thermal_id.train --processed-dir data/processed_t2 --n-train 400000 --epochs 10 --out-dir runs/cnn_t2_400k --resume
# variante hybride (CNN + 36 descripteurs) :
uv run python -m thermal_id.train --processed-dir data/processed_t2 --use-features --out-dir runs/hybride
```

Sur GPU (Kaggle, Colab), la même commande utilise CUDA automatiquement, avec précision mixte.

### Exécuter les notebooks

Ouvrir `notebooks/` dans VS Code ou Jupyter avec le noyau de `.venv`, ou les exécuter en ligne de commande :

```powershell
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_baseline.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03_cnn.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_resultats.ipynb
```

Les notebooks sont livrés déjà exécutés (sorties visibles). Pour les réexécuter, il faut d'abord les données
préparées (`data/processed/` et `data/processed_t2/`, étape 1 et première commande de la comparaison) ; le
notebook 4 relit en plus les résultats de `results/` et `runs/`, fournis dans l'archive.

Durées mesurées : `01_eda` ≈ 2 min, `02_baseline` ≈ 80 min (il réajuste la baseline et refait l'analyse à signal
fixé), `03_cnn` ≈ 30 min (dont le test de sur-apprentissage), `04_resultats` ≈ 1 min (il relit les résultats des
étapes 1 à 5).

Les chemins sont résolus par `thermal_id.paths` : notebooks et commandes fonctionnent quel que soit le dossier courant.

---

## 4. Protocole d'évaluation

| Découpage | Batches | Simulations | Usage |
|---|---|---|---|
| Train | 1 à 16 | 800 000 | ajustement des modèles et de la normalisation |
| Validation | 17 et 18 | 100 000 | choix du modèle (perte de validation, critère fixé à l'avance) |
| Test | 19 et 20 | 100 000 | une seule évaluation finale |

Le découpage se fait par batch, et non par tirage aléatoire de simulations, pour éviter toute fuite
d'information entre les ensembles ; l'EDA vérifie que chaque simulation correspond à un bâtiment distinct.

Métriques par cible, en unités physiques : MAE, MAPE et R². Le MAPE est aussi calculé par famille d'excitation
(sans ou avec chauffage, quintiles de `p_on` et de `min_duration`).

---

## 5. Points clés de l'analyse

- **Résolution du capteur.** Un CNN entraîné sur T à pleine précision s'effondre quand on l'évalue sur T arrondi à
  0,01 °C (R² HLC de 0,44 à 0,07) : il s'appuyait sur des détails invisibles pour un capteur réel. Réentraîné sur
  T arrondi, il retrouve presque toute sa performance (MAPE HLC 8,36 % contre 8,20 %). Un modèle doit être entraîné
  avec la résolution, et idéalement le bruit, du capteur réel.
- **Un plafond physique.** Le simulateur est linéaire, d'ordre 4 environ et sans bruit (un modèle ARX d'ordre 4
  reproduit chaque courbe à la précision de stockage près). Pourtant, l'identification directe du HLC échoue
  (erreur de plus de 100 %) : sur 8 h, un bâtiment qui perd beaucoup avec une grande inertie et un bâtiment qui perd
  peu avec une faible inertie produisent presque la même courbe. La limite vient de la durée de mesure, pas du
  modèle.
- **Lien avec la méthode de coEnergy.** Ce modèle correspond à la phase 1 de la méthode décrite dans l'article de
  référence (modèle généraliste). Les limites observées justifient les choix de l'article : signal de chauffage
  optimisé pour chaque test, modèle réentraîné sur la météo et le chauffage réellement mesurés, bruit capteur ajouté
  à l'entraînement.
- **Limites.** Données simulées d'une seule maison, sans météo variable, soleil, infiltrations ni bruit capteur :
  le modèle ne se transpose pas tel quel à un bâtiment réel. Détails et perspectives : notebook 4, sections 7 et 8.
