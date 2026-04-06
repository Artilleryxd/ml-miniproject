# ML Mini Project Web App

A simple Flask website that lets you upload a compatible classification CSV and evaluates:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes

## Run Locally

1. Create/activate a Python environment.
2. Install dependencies:

```bash
pip install -r app/requirements.txt
```

3. Start the app from project root:

```bash
python app/app.py
```

4. Open:

```text
http://127.0.0.1:5000
```

## Dataset Expectations

- CSV should include at least one feature column and one target column.
- If target column is not provided in the form, the last CSV column is used.
- Categorical features are auto-encoded with one-hot encoding.
- Rows with missing target values are removed automatically.
