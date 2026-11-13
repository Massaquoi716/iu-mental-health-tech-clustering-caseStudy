# Mental Health in Technology — Unsupervised ML Case Study

An end-to-end unsupervised machine learning case study analysing workplace mental health survey data from technology-oriented employees. The objective is to segment employees into interpretable, statistically validated personas and identify actionable, pre-emptive points of leverage for Human Resources (HR).

Developed for the course *Machine Learning — Unsupervised Learning and Feature Engineering* at **IU International University of Applied Sciences**.

---

## Project Goal & Methodology

Survey data in HR analytics is high-dimensional, dominated by categorical and conditional skip-logic fields, and prone to free-text noise. Rather than clustering directly on clinical diagnoses (which risks trivial "healthy vs. ill" splits), this project separates **workplace environment & stigma features** from **clinical and demographic outcomes**:

1. **Data Audit & Structural Forensics:** Identify deterministic survey skip logic (MNAR) across self-employed vs. employed branches, duplicate headers, pipe-delimited (`|`) multi-selects, and implausible outliers in `What is your age?`.
2. **Cohort Scoping & Cleaning (`n = 1,433` → `1,131`):** Filter strictly to employed technology workers (`Are you self-employed? == 0` AND (`Is your employer primarily a tech company/organization? == 1` OR `Is your primary role within your company related to tech/IT? == 1`)), median-impute extreme age typos (`3`, `99`, `323` → `32.0`), consolidate `60` free-text gender entries losslessly, and resolve conditional missingness semantically.
3. **Hybrid Feature Engineering (`22` Workplace Inputs → `61` Clean Features):** Map `10` monotonic ordinal/binary features to `[0, 1]`, One-Hot Encode `12` nominal/epistemic features (preserving `"I don't know"` as an explicit awareness signal), remove `100%` collinearity, and standardize via `StandardScaler`.
4. **Iterative Dimensionality Reduction & Clustering:**
   - **Baseline:** PCA + `K-Means` on the `62`-D matrix (diagnosed curse-of-dimensionality & rare-dummy artifacts).
   - **Refinement (`Iteration 2A`):** `K-Means` on the top `10` Principal Components (`42.14%` denoised variance), cross-validated against Ward Hierarchical clustering and **native mixed-data Gower Distance + K-Medoids (`PAM`)** (`ARI = 0.2439` at `k = 4`).
   - **Visualizations:** Global linear structure via **2D PCA** (`PC1`: Psychological Safety ↔ Stigma; `PC2`: Uncertainty ↔ HR Infrastructure) alongside local non-linear neighborhoods via **2D t-SNE**.
5. **Post-Hoc Statistical & Qualitative Profiling:** Validate cluster differences across held-out clinical disorders, treatment uptake, work interference (`Chi-Square`, `Kruskal-Wallis`, Cramér's $V$), specific diagnoses (`Mood`, `Anxiety`, `ADHD`, `PTSD`), and open-ended interview explanations (`TF-IDF` & verbatim quotes).

---

## Key Findings (`k = 4` Workforce Personas)

| Cluster | Share (`n = 1,131`) | Key Organizational Profile | Key Clinical / HR Takeaway |
| :--- | :---: | :--- | :--- |
| **C0: Early-Career / Startup Blind Spot** | `35.7%` (`404`) | Youngest (`median = 31.0`), `59.9%` Back-end Developers; `86.4%` unsure if anonymity is protected; `5.4%` formal MH discussions. | Lowest awareness & treatment (`48.0%`); needs Day-1 onboarding & asynchronous confidentiality guides. |
| **C1: Covered Yet Stigma-Wary** | `18.4%` (`208`) | `59.6%` large firms (`>500` staff), `75.5%` US; **`89.9%` have benefits**, yet `52.4%` fear career harm & `40.9%` saw unsupportive responses. | **Benefits alone do not build trust.** Highest `ADHD` (`13.9%`) & `PTSD` (`7.7%`); needs manager bias training. |
| **C2: High-Stigma / Unprotected Climate** | `24.9%` (`282`) | `91.1%` fear career harm, `59.9%` difficult medical leave, only `6.7%` supervisor comfort, `73.0%` unsure of anonymity. | `73.8%` current/possible disorder; **`7.1%` frequent work interference *even when effectively treated*** (~3× higher than other clusters). |
| **C3: Psychologically Safe Champions** | `21.0%` (`237`) | Highest remote flexibility (`26.6%`), highest leadership density (`24.5%` Team Leads), `93.7%` expect no negative consequences. | Lowest treated work interference (`2.5%`) and `10×` higher interview openness (`21.5%` vs. `2.1%` in C2); benchmark culture to scale. |

*(All held-out clinical and work-interference differences across clusters are statistically significant at $p < 0.001$.)*

---

## Repository Structure

```
.
├── data/
│   ├── raw/                                     # Source CSV (not tracked by git)
│   │   └── mental-health-in-tech-2016.csv
│   └── processed/                               # Generated clustered dataset (not tracked by git)
│       └── osmi_2016_tech_clustered_k4.csv
│
├── notebooks/
│   └── mental_health_in_tech_clustering.ipynb   # Complete end-to-end analysis notebook
│
├── src/                                         # Shared project utilities
│   ├── __init__.py
│   ├── io_utils.py                              # CSV I/O helpers
│   ├── paths.py                                 # Project-root-anchored path constants
│   └── plotting.py                              # Style config, colour palette, and plot helpers
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Dataset

- **Expected file name:** `mental-health-in-tech-2016.csv`
- **Target location:** `data/raw/mental-health-in-tech-2016.csv`
- **Raw size:** `1,433` respondents × `63` columns (`0` exact duplicate rows)

The dataset is the *OSMI Mental Health in Tech Survey 2016* (Open Sourcing Mental Illness, LTD), measuring workplace mental health policies, perceived stigma, and mental health prevalence across technology-related roles.

### How to get the data

Data files (`data/**/*.csv`, `data/**/*.xlsx`, `data/**/*.json`) are excluded from version control via `.gitignore`. Download the dataset manually using either option below:

1. **Primary source (Kaggle):**  
   https://www.kaggle.com/datasets/osmi/mental-health-in-tech-2016
2. **Alternative source (IU SharePoint mirror):**  
   https://iubhfs-my.sharepoint.com/:f:/g/personal/christian_mueller-kett_iu_org/ErsnsfeHni1EnQ3jV0L823YBqmyROZ3uwdy5Lj8j8jlFXA?e=obwNWc

**Instructions:**

1. Create `data/raw/` at the project root if it does not already exist.
2. Download the CSV file from either source above.
3. Rename it to **`mental-health-in-tech-2016.csv`** and place it inside `data/raw/`.

---

## How to Run

### Prerequisites

- Python 3.10 or newer
- `pip`
- Jupyter Notebook or JupyterLab

### Step 1 — Clone the repository

```bash
git clone https://github.com/Massaquoi716/iu-mental-health-tech-clustering-caseStudy.git
cd iu-mental-health-tech-clustering-caseStudy
```

### Step 2 — Place the dataset

Ensure `data/raw/mental-health-in-tech-2016.csv` exists:

```
ls -lh data/raw/mental-health-in-tech-2016.csv
```

### Step 3 — Create a virtual environment and install dependencies

**Linux / macOS:**

```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (PowerShell):**

```
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (cmd.exe):**

```
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Run the notebook

Launch Jupyter from the project root and run all cells top-to-bottom in `notebooks/mental_health_in_tech_clustering.ipynb`:

```
jupyter lab
```

Running the notebook top-to-bottom reproduces all diagnostic tables, PCA/t-SNE visualizations, statistical tests, and exports `osmi_2016_tech_clustered_k4.csv` into `data/processed/`.

---

## Development & Reproducibility Notes

- Shared path, I/O, and plotting helpers live in `src/` (`paths.py`, `io_utils.py`, `plotting.py`).
- Random seeds (`random_state=42`) are fixed across `PCA`, `KMeans`, `TSNE`, `K-Medoids`, and quote sampling.
- Raw and derived CSVs inside `data/raw/` and `data/processed/` are excluded from Git via `.gitignore` (`data/**/*.csv`) and regenerated locally from `data/raw/mental-health-in-tech-2016.csv`.

---

## License

Submitted as academic coursework for IU International University of Applied Sciences. The survey dataset is provided by Open Sourcing Mental Illness (OSMI) under its original terms. All analysis code and documentation are the author's own work.

---

## Author

**Massaquoi L. Dehmie**  
IU International University of Applied Sciences
