# Experiment Log (All Runs)

## EXP-001: Finalize project structure, checklist template, and implement baseline features

### Branch Edit Summary
1. **Folder Restructuring**:
   - Created missing `models` directory.
   - Created missing tracking docs (`EXPERIMENTS.md`, `RESULTS.md`, `Branch_Edit_Summary.md`, `experiment_log.md`).
   - Moved `scripts/Sprint_2_EDA.ipynb` to `eda.ipynb` at the project root to match the global template.
   - Deleted unused `app.py` since the app doesn't have an API.
   - Created placeholder file `src/model_training.py`.
2. **docs/checklist.md**:
   - Added specific preprocessing steps extraction: `Handle missing values` and `Remove duplicates`.
   - Converted the list items to valid Markdown task list syntax (`- [ ]` and `- [x]`).
   - Updated the empty boxes to be filled with checks (`[✓]`) for completed items and Xs (`[X]`) for features not present in this minimal project.
   - Checked off new EDA implementations recently added: `Global random seed`, `Save raw data statistics`, `Sampling`, `Label numeric categorical and time data`, `Feature engineer datetime`, and `Save clean data statistics`.
3. **src/config.py**:
   - Added a `RANDOM_SEED` variable to serve as the global random seed.
4. **src/data_preprocessing.py**:
   - Expanded `load_data` to support optional Pandas `sample()` with `random_state`.
   - Added `save_statistics` to save `df.describe()` outputs to `data/`.
   - Added `label_categorical` to explicitly convert columns to categorical dtype.
   - Added `engineer_datetime_features` to compute `is_weekend` and `is_morning_order` features based on available ordinal data.
5. **main.py**:
   - Integrated `numpy` and `random` seeds using the global seed.
   - Injected the new data preprocessing tools into the main pipeline flow.
6. **tests/test_data_preprocessing.py**:
   - Created `tests` directory and added unit tests for `fill_missing`, `remove_duplicates`, `label_categorical`, and `engineer_datetime_features`.
   - Made the project unit testable and verified passage using `pytest`.
7. **README.md**:
   - Migrated from a template layout to a finalized canonical documentation format.
   - Reorganized the `Key Insights`, `Overview`, and specific workflow instructions.
   - Removed placeholder template tags and inapplicable API instructions.
   - Updated the directory tree output to correctly reflect the presence of the `scripts` folder and absence of `app.py`.

### Results
Since this is an Exploratory Data Analysis project, there are no machine learning model evaluations. However, the initial folder structure has successfully been consolidated according to the project specifications.

**Key Analytical Findings**:
- The peak times for customers to order are on Sunday and Monday (Days 0 and 1).
- Most orders occur between 10 AM and 4 PM.
- Reordering patterns highlight high frequency with certain item categories, validating data distribution consistency.
