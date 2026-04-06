# App Test Scenarios

Use these datasets with the upload form at `http://127.0.0.1:5000`.

## 1) Happy Path Binary Classification
- Dataset: `binary_clean.csv`
- Target Column: `Purchased`
- Drop Columns: `User ID`
- Test Size: `0.25`
- KNN Neighbors: `3`
- Expected:
  - Analysis succeeds.
  - All 3 models are shown.
  - Confusion matrix and classification report render for each model.
  - Best model summary appears at the top.

## 2) Noisy + Missing Data Robustness
- Dataset: `binary_noisy_missing.csv`
- Target Column: `Purchased`
- Drop Columns: (leave empty)
- Expected:
  - Analysis succeeds.
  - Rows with missing target are removed automatically.
  - Metrics are lower than `binary_clean.csv` due to noise/missing data.

## 3) Multiclass Target Behavior
- Dataset: `multiclass_flowers.csv`
- Target Column: `species`
- Expected:
  - Analysis succeeds with multiclass output.
  - ROC-AUC appears as `N/A (multiclass)`.
  - Confusion matrix is 3x3.

## 4) Validation Error: Single-Class Target
- Dataset: `single_class_error.csv`
- Target Column: `Purchased`
- Expected:
  - App returns an error message:
    - `Target column must have at least two classes.`

## 5) Validation Error: Wrong Target Name
- Dataset: `tiny_binary.csv`
- Target Column: `Purchased` (intentionally wrong)
- Expected:
  - App returns an error message:
    - `Target column 'Purchased' was not found in uploaded CSV.`

## 6) Parameter Validation: Invalid KNN
- Dataset: `binary_clean.csv`
- Target Column: `Purchased`
- KNN Neighbors: `0`
- Expected:
  - App returns an error message:
    - `KNN neighbors must be a positive integer.`

## 7) Parameter Validation: Invalid Test Size
- Dataset: `binary_clean.csv`
- Target Column: `Purchased`
- Test Size: `0.8`
- Expected:
  - App returns an error message:
    - `Test size must be between 0.1 and 0.5.`
