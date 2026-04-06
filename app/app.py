from __future__ import annotations

from io import StringIO

import pandas as pd
from flask import Flask, render_template, request

from ml_analysis import AnalysisError, run_classification_analysis

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


@app.route("/", methods=["GET"])
def index() -> str:
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze() -> str:
    uploaded_file = request.files.get("dataset")
    target_column = request.form.get("target_column", "").strip()
    drop_columns = request.form.get("drop_columns", "").strip()
    test_size = request.form.get("test_size", "0.25").strip()
    knn_neighbors = request.form.get("knn_neighbors", "3").strip()
    random_state = request.form.get("random_state", "42").strip()

    if uploaded_file is None or uploaded_file.filename == "":
        return render_template("index.html", error="Please upload a CSV file.")

    if not uploaded_file.filename.lower().endswith(".csv"):
        return render_template("index.html", error="Only .csv files are supported.")

    try:
        file_content = uploaded_file.read().decode("utf-8", errors="ignore")
        dataframe = pd.read_csv(StringIO(file_content))
    except Exception as exc:  # pragma: no cover
        return render_template("index.html", error=f"Could not read CSV: {exc}")

    try:
        test_size_float = float(test_size)
        knn_neighbors_int = int(knn_neighbors)
        random_state_int = int(random_state)

        drop_columns_list = [c.strip() for c in drop_columns.split(",") if c.strip()]

        results = run_classification_analysis(
            dataframe=dataframe,
            target_column=target_column if target_column else None,
            drop_columns=drop_columns_list,
            test_size=test_size_float,
            knn_neighbors=knn_neighbors_int,
            random_state=random_state_int,
        )
    except ValueError:
        return render_template(
            "index.html",
            error=(
                "Test size must be a number, and KNN neighbors/random state must be integers."
            ),
        )
    except AnalysisError as exc:
        return render_template("index.html", error=str(exc))

    return render_template("results.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)
