# Module 2 — Analytics (`/analytics`)

Profiles the Titanic dataset, cleans it defensibly, tells a visual data
story, then builds and rigorously evaluates a full predictive-modeling
pipeline on top of the same cleaned data.

## Pipeline stages

| Script | Input | Output | Purpose |
|---|---|---|---|
| `eda.py` | `sns.load_dataset(''titanic'')` (network/cache, loaded once) | `titanic.csv`, `charts/*.png`, `eda_output.txt` | Profile, clean, univariate/bivariate/correlation/multivariate story |
| `modeling.py` | `titanic.csv` (no second raw load) | `best_pipeline.joblib`, more `charts/*.png`, `modeling_output.txt` | Stratified split, leak-free preprocessing, 3 classifiers, imbalance comparison, tuning, regression side-task, final comparison + saved pipeline |

## Install & run

From the `analytics/` folder, with a virtual environment active:

```powershell
pip install seaborn scikit-learn imbalanced-learn joblib pandas matplotlib
python eda.py > eda_output.txt
python modeling.py > modeling_output.txt
```

`sns.load_dataset(''titanic'')` is called **exactly once**, inside `eda.py`.
It fetches the dataset from Seaborn''s online repository on first run and
caches it locally; subsequent runs reuse the cache. `titanic.csv` -- saved
immediately after that one load -- is the committed offline fallback, so
the module can be graded via `pd.read_csv("titanic.csv")` even without
network access. `modeling.py` reads only from `titanic.csv`; it never
calls `sns.load_dataset(...)` again.

## Part A -- Profiling, cleaning, and the data story

### Missing values

| Column | Missing % | Strategy | Justification |
|---|---|---|---|
| `deck` | 77.22% | **Dropped column** | Far too high to impute reliably -- would be mostly fabricating values. Also largely redundant with `pclass`/`fare` (cabin deck correlates with class), so little unique signal is lost. |
| `age` | 19.87% | **Imputed** (median = 28.0) | Within the 5-30% band per the threshold rule. Median chosen over mean because `age`/`fare`-type distributions are typically skewed and the median is more robust to skew. |
| `embarked` / `embark_town` | 0.22% (2 rows) | **Dropped those rows** | Under the 5% threshold -- dropping 2 rows from 891 has negligible impact on the dataset, and is simpler/safer than imputing a categorical port of embarkation. |

After cleaning: 889 rows, 0 missing values remaining.

### Univariate analysis (age, fare)

- **IQR outliers:** `age` -> **65 outliers** (bounds [2.50, 54.50]); `fare` -> **114 outliers** (bounds [-26.76, 65.66]).
- **`fare` central tendency:** mean = 32.10, median = 14.45, mode = 8.05.
  Since **mean > median > mode**, `fare` is **right-skewed** -- a long tail
  of a small number of very high fares pulls the mean well above the
  median, while most passengers paid comparatively little.
- Charts: `charts/univariate_age.png`, `charts/univariate_fare.png` (histogram + box plot for each).

### Bivariate analysis: survival rate

- **By sex:** female 74.0% vs. male 18.9% -- a roughly 4x gap. Sex is clearly one of the strongest survival predictors, consistent with a "women and children first" evacuation pattern.
- **By class:** class 1 = 62.6%, class 2 = 47.3%, class 3 = 24.2% -- survival rate drops steadily with class, likely reflecting cabin location relative to lifeboats and boarding priority.
- **By sex + class combined:** female/class 1 = 96.7% (highest) down to male/class 3 = 13.5% (lowest) -- the two factors compound: being female and wealthy gave by far the best odds, being male and poor gave the worst.

### Correlation heatmap (6 columns)

Computed on exactly `survived, pclass, age, sibsp, parch, fare` (see `charts/correlation_heatmap.png`). The two strongest off-diagonal correlations:

1. **`pclass` <-> `fare`: -0.548** -- lower class number (i.e. higher class) strongly associates with higher fare, as expected (class 1 tickets cost more).
2. **`sibsp` <-> `parch`: 0.415** -- passengers traveling with more siblings/spouses also tend to travel with more parents/children, i.e. family size components move together (families travel as units).

### Multivariate data story (4 charts)

1. **`multivariate_1_survival_by_class_sex.png`** -- Grouped bar of survival rate by class and sex. Female survival stays high (>90%) across classes 1-2 and only drops to 50% in class 3, while male survival is low across all classes and collapses further in class 3. This shows sex was the dominant factor, with class acting as a secondary modifier -- especially harmful for men in steerage.
2. **`multivariate_2_age_by_survival.png`** -- Box plot of age by survival outcome. Survivors skew slightly younger, with a visibly lower median age than non-survivors, and both groups show a wide spread with several older outliers. This is mild evidence for age-based prioritization (e.g. children), though the effect is smaller than the sex/class effects.
3. **`multivariate_3_fare_vs_age.png`** -- Scatter of fare vs. age colored by survival. Survivors (orange) cluster more densely at higher fares across most ages, while non-survivors (blue) dominate the low-fare band. This visually reinforces the `pclass`/`fare` correlation with survival -- passengers who paid more had a real, visible edge.
4. **`multivariate_4_survival_by_family_size.png`** -- Bar of survival rate by family size (`sibsp` + `parch`). Passengers traveling completely alone (family size 0) have a lower survival rate than those with small families (1-3), but very large families (6+) fare worst of all. This suggests a "sweet spot" -- some companionship helped (e.g. mutual aid boarding lifeboats), but large families may have been harder to evacuate together.

### Exploratory standardization check (EDA-only)

Z-score standardization was applied to `age` and `fare` as a sanity check only -- **not** used in the modeling pipeline (which fits its own scaler on the training split):

| Column | Before mean | Before std | After mean | After std |
|---|---|---|---|---|
| `age` | 29.32 | 12.98 | 0.0000 | 1.0000 |
| `fare` | 32.10 | 49.70 | 0.0000 | 1.0000 |

Both columns land at (approximately) mean 0, std 1 after transformation, confirming the z-score formula was applied correctly.

## Part B -- Predictive modeling

### Stratified split

Split 80/20 with `stratify=y`. **Justification:** `survived` is moderately
imbalanced (~62% did not survive / 38% survived). A plain random split
risks producing a test set with a noticeably different survival rate than
the training set purely by chance, distorting every downstream metric --
stratifying preserves the same class ratio in both splits, giving a fairer
evaluation. (Confirmed: train survival rate 0.3826, test 0.3820 -- very close.)

### Preprocessing (leak-free)

A `ColumnTransformer` handles numeric features (`pclass, age, sibsp, parch, fare`
-> median-impute + `StandardScaler`) and categorical features (`sex, embarked`
-> most-frequent-impute + `OneHotEncoder`), wrapped in a `Pipeline` with the
final estimator. Every step is fit only via `.fit(X_train, y_train)` -- the
test split only ever sees `.transform()`. `alive` (a literal text duplicate
of the target) and other redundant/derived columns (`class`, `who`,
`adult_male`, `alone`, `embark_town`) are excluded from the feature set
entirely to avoid leakage and duplicate signal.

### Classifier comparison

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.809 | 0.783 | 0.691 | 0.734 | 0.861 |
| Decision Tree | 0.764 | 0.760 | 0.559 | 0.644 | 0.837 |
| Random Forest | 0.809 | 0.766 | 0.721 | 0.742 | 0.820 |
| **Random Forest (tuned)** | **0.837** | **0.882** | 0.662 | **0.756** | 0.839 |

Decision tree visualized with labeled features/classes: `charts/decision_tree.png`.
ROC curves for all three base classifiers: `charts/roc_curves.png`.

### Imbalance handling comparison (Logistic Regression)

Overall class balance: 61.75% not-survived / 38.25% survived.

| Strategy | Precision | Recall | F1 |
|---|---|---|---|
| Baseline (no handling) | 0.783 | 0.691 | 0.734 |
| `class_weight=''balanced''` | 0.718 | 0.750 | 0.734 |
| **SMOTE (train fold only)** | 0.735 | 0.735 | **0.735** |

**Conclusion:** SMOTE gave a marginally better F1 than the alternatives,
though the gap is small since the imbalance here (~62/38) is moderate
rather than severe. `class_weight=''balanced''` traded precision for higher
recall; SMOTE landed in between. For a use case where missing an actual
survivor prediction is costlier than a false alarm, the recall gains from
`class_weight`/SMOTE over the baseline are worth the small precision cost.

### Hyperparameter tuning (GridSearchCV)

Tuned `RandomForestClassifier(oob_score=True, bootstrap=True)` over
`n_estimators in {100,200,300}`, `max_depth in {None,5,10}`,
`max_features in {''sqrt'',''log2''}`, 5-fold CV.

- **Best params:** `max_depth=5, max_features=''sqrt'', n_estimators=300`
- **Best CV accuracy:** 0.820
- **OOB score:** 0.821

The OOB score closely tracks the held-out test accuracy (0.837), suggesting the tuned model isn''t overfitting to the training split.

### Regression side-task: predicting `fare`

Multivariate linear regression predicting `fare` from `pclass, age, sibsp,
parch, sex, embarked, survived`:

- **MAE:** 21.10
- **RMSE:** 41.70
- **R2:** 0.348
- **Adjusted R2:** 0.309 (n=178, p=10)

**Residual plot:** `charts/regression_residuals.png`. The residual spread
visibly widens as predicted fare increases (a funnel/cone shape rather than
a uniform band) -- this is **heteroscedastic**. This matches the right-skew
found in `fare` during EDA: a handful of very high fares are inherently
harder to predict precisely, producing larger residuals at the high end.

### Final comparison and recommendation

**Classification metrics** (one scale):

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.809 | 0.783 | 0.691 | 0.734 | 0.861 |
| Decision Tree | 0.764 | 0.760 | 0.559 | 0.644 | 0.837 |
| Random Forest | 0.809 | 0.766 | 0.721 | 0.742 | 0.820 |
| Random Forest (tuned) | 0.837 | 0.882 | 0.662 | 0.756 | 0.839 |

**Regression metrics** (a separate scale -- not directly comparable to the classification metrics above):

| Model | MAE | RMSE | R2 | Adjusted R2 |
|---|---|---|---|---|
| Linear Regression (fare) | 21.10 | 41.70 | 0.348 | 0.309 |

**Recommendation:** Deploy the **tuned Random Forest**. It achieved the
highest F1 (0.756) and accuracy (0.837) among all models tested, and its
precision (0.882) is notably strong -- meaning when it predicts survival,
it''s usually right. Its OOB score (0.821) closely tracks its held-out test
accuracy, indicating the hyperparameter search didn''t overfit to the
training data. While plain Logistic Regression has a slightly higher AUC
(0.861 vs 0.839), the tuned Random Forest''s better overall F1/accuracy
balance and its use as the saved, deployable pipeline make it the more
practical choice for this classification task.

### Saved pipeline

The complete fitted pipeline (`ColumnTransformer` preprocessing + tuned
Random Forest estimator together) is saved via `joblib.dump(...)` to
`best_pipeline.joblib`. `modeling.py` demonstrates reloading it with
`joblib.load(...)` and predicting on a **raw, unpreprocessed** sample row
(including a deliberately missing `age` value) -- confirming the saved
artifact is fully self-contained and usable end-to-end on new raw data,
with no external preprocessing required before calling `.predict()`.

## Repository files

- `eda.py`, `modeling.py` -- pipeline scripts
- `titanic.csv` -- the one committed offline fallback (cleaned dataset)
- `eda_output.txt`, `modeling_output.txt` -- full console output of both scripts
- `charts/` -- all 10 saved chart PNGs (2 univariate, 1 correlation heatmap, 4 multivariate, ROC curves, decision tree, regression residuals)
- `best_pipeline.joblib` -- the saved, deployable, fully fitted pipeline
