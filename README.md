# Mars Surface Image Classification

Classifying geological landmarks on the surface of Mars from NASA HiRISE orbital imagery, using classical machine learning (SVM) and deep convolutional neural networks (AlexNet and GoogLeNet), with a focus on handling **severe class imbalance**.

This was a university group project. It compares oversampling strategies (Random Oversampling vs. ADASYN) and two CNN architectures on a real, heavily imbalanced remote-sensing dataset.

## Problem

The [HiRISE](https://www.uahirise.org/) camera aboard NASA's Mars Reconnaissance Orbiter produces high-resolution images of the Martian surface. The task is to classify image tiles into **8 landmark classes**:

| Label | Class | Label | Class |
|------:|-------|------:|-------|
| 0 | other | 4 | bright dune |
| 1 | crater | 5 | impact ejecta |
| 2 | dark dune | 6 | swiss cheese |
| 3 | slope streak | 7 | spider |

The central challenge is **extreme class imbalance**: of the 10,815 image tiles, ~81% belong to class `other`, while rare classes such as `impact ejecta` have fewer than 75 examples.

## Dataset

The data is the publicly released NASA JPL *Mars orbital image (HiRISE) labeled data set, version 3.2* (Wagstaff et al., "Deep Mars"). It is included directly in this repository:

- **`map-proj-v3_2/`** — 10,815 grayscale image tiles (227×227), JPEG.
- **`labels-map-proj_v3_2.txt`** — image filename → class label.
- **`labels-map-proj_v3_2_train_val_test.txt`** — official train/validation/test split (includes augmented filename variants).
- **`landmarks_map-proj-v3_2_classmap.csv`** — label index → class name mapping.

> Filenames ending in `-fv`, `-fh`, `-brt`, `-r90`, `-r180`, `-r270` are flipped/brightened/rotated augmentations of base tiles and are filtered out during preprocessing.

## Approach

### Preprocessing & splitting
- Images loaded as grayscale, resized to **227×227**, normalized to `[0, 1]`.
- Stratified **60% train / 10% validation / 30% test** split.
- ~4,500 majority-class (`other`) training samples removed to reduce the imbalance before resampling.

### Handling class imbalance
Two oversampling strategies (from [`imbalanced-learn`](https://imbalanced-learn.org/)) are compared, each rebalancing every minority class up to the majority count:
- **Random Oversampling** — duplicates existing minority samples.
- **ADASYN** — synthesizes new minority samples adaptively in feature-dense regions.

A 7-fold image **augmentation** (rotations 90/180/270°, horizontal/vertical flips, zoom) is then applied to the training set only.

### Models
| Model | Library | Notes |
|-------|---------|-------|
| **SVM** | scikit-learn | On flattened pixels; grid search over `C ∈ {0.1,1,10}`, kernel ∈ {linear, rbf, poly}, gamma ∈ {0.1,1,scale}; 3-fold CV. Best: RBF, `C=10`, `gamma=scale`. |
| **AlexNet** | TensorFlow/Keras | 5 conv blocks + 2×4096 dense, dropout 0.5, softmax(8). |
| **GoogLeNet** | TensorFlow/Keras | Inception modules + global average pooling, softmax(8). |

CNNs trained with **SGD** (momentum 0.9, Nesterov, weight decay 5e-4), learning rates `{0.005, 0.01}`, batch sizes `{32, 128}`, up to 30 epochs.

## Results

Test-set accuracy (3,028 images) for the final 30-epoch CNN runs:

| Model | Imbalanced | Random Oversampling | ADASYN |
|-------|:----------:|:-------------------:|:------:|
| **AlexNet**   | 0.88 | 0.80 | 0.82 |
| **GoogLeNet** | 0.89 | 0.86 | 0.85 |

> ⚠️ **Read these numbers carefully.** Because ~37% of the test set is the majority `other` class, raw accuracy is misleading — the "Imbalanced" models score highest simply by favoring the majority class. The oversampled (Random/ADASYN) models trade a few points of overall accuracy for substantially better **per-class recall on rare landmarks**, which is the metric that actually matters here. Per-class precision/recall/F1 and confusion matrices for every configuration are in the result notebooks.

The CNNs vastly outperform the SVM baseline, whose test accuracy collapsed on the rare classes (≈0.18 on balanced training) — illustrating why deep features are needed for this task.

## Repository structure

```
.
├── 00_Data_Preparation.ipynb        # Parse/clean metadata, build the labels table
├── 01_Support_Vector_Machine.ipynb  # SVM baseline (grid search + evaluation)
├── 02_Neural_Network.ipynb          # AlexNet & GoogLeNet (main deliverable)
│
├── map-proj-v3_2/                   # HiRISE image tiles (dataset)
├── labels-map-proj_v3_2.txt
├── labels-map-proj_v3_2_train_val_test.txt
├── landmarks_map-proj-v3_2_classmap.csv
│
├── experiments/                     # Exploratory notebooks
│   ├── augmentations.ipynb          # Image augmentation experiments
│   ├── oversampling_comparison.ipynb
│   └── z0{1..4}_*.ipynb             # Per-model × per-oversampling experiments
│
└── Results/
    ├── final-models/                # The six final 30-epoch runs (per table above)
    └── grid-search/                 # Hyperparameter sweep (model × oversampling × bs × lr)
```

## Running the notebooks

The notebooks were developed in **Google Colab** (GPU runtime) and read the dataset from the repository paths above. To run locally:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Training the CNNs requires a GPU and several hours; the committed notebooks already contain their outputs (metrics, plots, confusion matrices) so the results are viewable without re-running.

## Team

Developed as a group project by **Finn Feddersen**, **Leonard Brenk** ([@leobreo](https://github.com/leobreo)), and [@felix1102](https://github.com/felix1102).

## Acknowledgements

Dataset courtesy of NASA/JPL-Caltech and the *Deep Mars* effort (Wagstaff et al.). The HiRISE imagery is publicly available for research and educational use.
