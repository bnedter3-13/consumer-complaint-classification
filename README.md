# Consumer Complaint Classification
### 10 classical ML algorithms vs. LoRA fine-tuned DistilBERT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bnedter3-13/consumer-complaint-classification/blob/main/ML_Project_Notebook.ipynb)

Multi-class text classification: given a consumer complaint (free text plus a few metadata fields), predict the financial **Product** it is about (13 classes).
Data: the U.S. CFPB Consumer Complaint Database ([Kaggle: `selener/consumer-complaint-database`](https://www.kaggle.com/datasets/selener/consumer-complaint-database)).
Goal: beat the public Kaggle benchmark of **77.9 %** accuracy (LinearSVC, 5-fold CV on a 10,000-row sample).

Course project — Machine Learning, University of Hail. Author: Bandar Al-Shammari.

## Result

| | |
|---|---|
| **Best model** | LoRA-fine-tuned DistilBERT |
| **Test accuracy** | **88.93 %** (macro-F1 0.763) |
| **Kaggle benchmark** | 77.9 % → **+11.03 points** |
| **Best classical model** | Logistic Regression, 88.54 % |

Test set: 73,389 complaints that no model saw during training or tuning. Every model uses the same train rows and the same test rows.

| Model | Accuracy | Macro-F1 | Train time (s) | Features |
|---|---:|---:|---:|---|
| **LoRA DistilBERT** | **0.8893** | 0.7632 | 1,190 | text + metadata tokens |
| Logistic Regression | 0.8854 | 0.7439 | 763 | sparse TF-IDF + one-hot |
| Linear SVC (tuned) | 0.8841 | 0.7458 | 65 | sparse |
| Linear SVC | 0.8836 | 0.7499 | 68 | sparse |
| Logistic Regression (tuned) | 0.8798 | 0.7584 | 763 | sparse |
| XGBoost | 0.8654 | 0.7127 | 597 | SVD 300 |
| Random Forest | 0.8262 | 0.6003 | 383 | SVD 300 |
| Gradient Boosting (histogram-based) | 0.8261 | 0.6384 | 61 | SVD 300 |
| Extra Trees | 0.8161 | 0.5793 | 53 | SVD 300 |
| Multinomial Naive Bayes | 0.7580 | 0.3582 | 1 | sparse |
| K-Nearest Neighbors | 0.7096 | 0.4863 | 0.4 | SVD 300 |
| Decision Tree | 0.7013 | 0.4799 | 920 | SVD 300 |
| AdaBoost | 0.6983 | 0.5066 | 9,414 | SVD 300 |
| *Linear SVC, text only (ablation)* | *0.8352* | *0.6524* | *67* | *text only* |

Seven of the eleven algorithms beat the benchmark. Full table: [`results_summary.csv`](results_summary.csv).

### Where does the improvement over Kaggle come from?

| Step | Accuracy | Gain |
|---|---:|---:|
| Kaggle benchmark (10K rows, 5-fold CV) | 77.90 % | |
| Linear SVC, text only, our split | 83.52 % | +5.62 (much more training data) |
| Linear SVC, text + metadata columns | 88.36 % | +4.84 (metadata columns) |
| LoRA DistilBERT, text + metadata | 88.93 % | +0.57 (fine-tuned transformer) |

Most of the gain comes from using all the data and the metadata columns, not from the choice of algorithm. A plain Linear SVC reaches 88.36 % in about one minute of training, versus about 20 minutes for LoRA. The step sizes are a rough breakdown: the benchmark was measured with 5-fold CV on 10K rows, ours on a held-out test set. The notebook also reproduces the benchmark itself: 77.9 % with its protocol, 77.8 % in a leak-free variant.

## Data

| | Rows | Columns |
|---|---:|---:|
| Full file (`rows.csv`) | 1,282,355 | 18 |
| With a complaint narrative | 383,564 | |
| Used (duplicate texts removed) | **366,945** | **10** (9 inputs + target) |

- **Split:** stratified 80 / 20, seed 42 → 293,556 train / 73,389 test. No subsampling.
- **Classes:** 13, following the Kaggle notebook's mapping so the comparison is fair.
- **Duplicates:** 16,619 complaints with identical text were removed *after* merging classes and *before* splitting. The notebook checks that no test text also appears in the training set (result: 0).
- **Inputs:** `Consumer complaint narrative` (text), `Company`, `State`, `Submitted via`, `Tags`, `Consumer consent provided?`, `Company response to consumer`, `Timely response?`, `Date received` (→ year, month). **Target:** `Product`.
- **Left out on purpose:** `Sub-product`, `Issue`, `Sub-issue` (they leak the label), `Complaint ID`, `ZIP code`, `Date sent to company`, `Company public response`, `Consumer disputed?`.

## Method

- **Text features:** TF-IDF (unigrams + bigrams, up to 300K features), fitted on the training rows only.
- **Metadata:** one-hot encoded and appended to the sparse matrix. Its weight was chosen on a validation split taken from the training rows.
- **Linear models and Naive Bayes** use the sparse matrix. **Trees, KNN, AdaBoost, XGBoost and Gradient Boosting** use a 300-dimensional TruncatedSVD projection.
- **Gradient Boosting** is scikit-learn's `HistGradientBoostingClassifier` (the classic version is optional, `USE_CLASSIC_GB`).
- **Tuning:** `RandomizedSearchCV` (3-fold) on 60K training rows for Linear SVC (15 settings) and Logistic Regression (8 settings), then refit on all training rows. It changed little: Linear SVC 88.36 → 88.41, Logistic Regression 88.54 → 87.98.
- **LoRA:** `distilbert-base-uncased`, rank 16, alpha 32, dropout 0.1, on the attention query/value layers. 895,501 trainable parameters of 67.9M (1.3 %, adapters plus the classification head). Input is the complaint text plus the metadata as a sentence pair (max 256 tokens). 2 epochs, batch 64, learning rate 2e-4, bf16 on an NVIDIA L4. About 20 minutes.
- **Reproducibility:** results are cached on Drive with a signature of the settings. A saved result is only reused if it was produced with exactly the same settings, and training resumes from checkpoints after a disconnect.

## How to run

1. Download `rows.csv` from the Kaggle dataset and put it in `MyDrive/ML_Project/` on Google Drive.
2. Open the notebook in Colab (badge above). Colab Pro with High-RAM is recommended for the full run.
3. First time only: set `QUICK_TEST = True` in the config cell and run everything once (a few minutes) to check that nothing breaks, then set it back to `False`.
4. **Sections 0 → 6** (data, EDA, features, benchmark, ten classical models, tuning): CPU + High-RAM runtime. The full run takes several hours, almost all of it AdaBoost (about 2.6 hours).
5. **Section 7 (LoRA):** switch to a GPU runtime (L4, or T4). The runtime restarts, so re-run only Section 0 and Section 1, then Section 7. The classical results are loaded from Drive.
6. **Section 8:** builds the final comparison from every saved result. Needs Sections 0 and 1 only.

If Colab disconnects, re-run Sections 0 and 1 and continue. Finished models are skipped.

Troubleshooting: if `get_peft_model` fails with a `torchao` version error, run `!pip uninstall -y torchao` and restart the session.

## Limitations

- One train/test split and one seed. The 0.4 to 0.6 point gap between LoRA and the best linear models was not tested for statistical significance.
- `Company response to consumer` and `Timely response?` are only known after a complaint has been handled. They help accuracy but would not exist at the moment a new complaint arrives.
- The benchmark comparison is approximate: the benchmark used 5-fold CV on 10K rows, this project a held-out test set on the full data.
- Logistic Regression (`lbfgs`) stops at its iteration limit without fully converging (a `ConvergenceWarning`). The result is valid but could change slightly with more iterations.
- Macro-F1 is much lower than accuracy for several models, because the classes are imbalanced and small classes are harder.

## Repository contents

| File | What it is |
|---|---|
| `ML_Project_Notebook.ipynb` | The whole project: data, EDA, features, 10 classical models, tuning, LoRA, final comparison |
| `results_summary.csv` | Final results table |
| `ML_Project_Presentation.pdf` / `.pptx` | Project presentation |

`rows.csv`, feature caches and LoRA checkpoints are not included (size). They are rebuilt by the notebook.
