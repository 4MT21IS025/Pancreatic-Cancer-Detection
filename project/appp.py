
import os
import io
import base64
from pathlib import Path

import pandas as pd
import matplotlib

# Use a non-interactive backend for Flask
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from flask import (
    Flask,
    render_template,
    request,
    session,
    redirect,
    url_for,
)
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# --------------------------------------------------
# 1. FLASK APPLICATION CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)

app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "development-only-change-this-secret"
)


# --------------------------------------------------
# 2. LOAD AND PREPROCESS DATASET
# --------------------------------------------------

dataset_path = BASE_DIR / "Debernardi et al 2020 data.csv"

if not dataset_path.exists():
    raise FileNotFoundError(
        f"Dataset not found: {dataset_path}\n"
        "Please place 'Debernardi et al 2020 data.csv' "
        "inside the project folder."
    )

df = pd.read_csv(dataset_path)

# Convert sex column into numeric values
df["sex"] = LabelEncoder().fit_transform(df["sex"])

# Fill missing values
df["plasma_CA19_9"] = df["plasma_CA19_9"].fillna(
    df["plasma_CA19_9"].mean()
)

df["REG1A"] = df["REG1A"].fillna(
    df["REG1A"].mean()
)

# Define column order
column_order = [
    "age",
    "sex",
    "diagnosis",
    "plasma_CA19_9",
    "creatinine",
    "LYVE1",
    "REG1B",
    "TFF1",
    "REG1A",
]

df = df[column_order]

# Define features and target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]


# --------------------------------------------------
# 3. TRAIN RANDOM FOREST CLASSIFIER
# --------------------------------------------------

clf = RandomForestClassifier(
    n_estimators=40,
    max_depth=6,
    random_state=42
)

clf.fit(X, y)


# --------------------------------------------------
# 4. GENERATE ANALYSIS CHART
# --------------------------------------------------

def create_chart(user_data):

    # Illustrative reference ranges from original code.
    # These are not validated clinical reference intervals.
    normal_ranges = {
        "age": (20, 60),
        "sex": (0, 1),
        "plasma_CA19_9": (0, 37),
        "creatinine": (0.6, 1.2),
        "LYVE1": (10, 50),
        "REG1B": (5, 30),
        "TFF1": (5, 25),
        "REG1A": (10, 40),
    }

    # Exclude sex from chart
    user_data = {
        key: value
        for key, value in user_data.items()
        if key != "sex"
    }

    normal_ranges = {
        key: value
        for key, value in normal_ranges.items()
        if key != "sex"
    }

    parameters = list(user_data.keys())
    user_values = list(user_data.values())

    normal_values = [
        sum(normal_ranges[key]) / 2
        for key in parameters
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.35
    indices = range(len(parameters))

    ax.bar(
        indices,
        user_values,
        bar_width,
        label="User Data",
        color="blue"
    )

    ax.bar(
        [i + bar_width for i in indices],
        normal_values,
        bar_width,
        label="Reference Midpoint (Illustrative)",
        color="green"
    )

    ax.set_xlabel("Parameters")
    ax.set_ylabel("Values")
    ax.set_title("User Data vs Illustrative Reference Values")

    ax.set_xticks([
        i + bar_width / 2
        for i in indices
    ])

    ax.set_xticklabels(
        parameters,
        rotation=45
    )

    ax.legend()

    img = io.BytesIO()

    plt.tight_layout()
    fig.savefig(img, format="png")
    img.seek(0)

    chart_img = base64.b64encode(
        img.getvalue()
    ).decode("utf-8")

    plt.close(fig)

    return chart_img


# --------------------------------------------------
# 5. HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():
    return render_template(
        "index.html",
        show_default_menu=True
    )


# --------------------------------------------------
# 6. ABOUT PAGE
# --------------------------------------------------

@app.route("/about")
def about():
    return render_template(
        "about.html",
        show_default_menu=True
    )


# --------------------------------------------------
# 7. FAQ PAGE
# --------------------------------------------------

@app.route("/faq")
def faq():
    return render_template(
        "faq.html",
        show_default_menu=True
    )


# --------------------------------------------------
# 8. PREDICTION PAGE
# --------------------------------------------------

@app.route("/prediction", methods=["GET", "POST"])
def prediction():

    explanation = ""
    higher_values = []

    if request.method == "POST":

        # Collect form inputs
        input_data = {
            "age": float(request.form["age"]),
            "sex": int(request.form["sex"]),
            "plasma_CA19_9": float(
                request.form["plasma_CA19_9"]
            ),
            "creatinine": float(
                request.form["creatinine"]
            ),
            "LYVE1": float(request.form["LYVE1"]),
            "REG1B": float(request.form["REG1B"]),
            "TFF1": float(request.form["TFF1"]),
            "REG1A": float(request.form["REG1A"]),
        }

        # Save user data for analysis
        session["user_data"] = input_data

        # Convert input into DataFrame
        input_df = pd.DataFrame([input_data])

        # Align columns with training data
        input_df = input_df[X.columns]

        # Predict diagnosis
        predicted_class = int(clf.predict(input_df)[0])

        # Illustrative thresholds from original project
        thresholds = {
            "age": 60,
            "plasma_CA19_9": 37,
            "creatinine": 1.2,
            "LYVE1": 50,
            "REG1B": 30,
            "TFF1": 25,
            "REG1A": 40,
        }

        # Identify values above illustrative thresholds
        for feature, value in input_data.items():

            if (
                feature in thresholds
                and value > thresholds[feature]
            ):
                higher_values.append(
                    f"{feature} ({value:.2f})"
                )

        # Prediction explanations
        if predicted_class == 1:

            explanation = (
                "The model predicts the No Cancer category. "
                "This result does not rule out pancreatic cancer. "
                "Consult a qualified healthcare professional "
                "for interpretation of your test results."
            )

        elif predicted_class == 2:

            explanation = (
                "The model predicts the Non-Cancerous "
                "Pancreatitis category. This is not a "
                "confirmed diagnosis. Please consult a "
                "qualified healthcare professional."
            )

        elif predicted_class == 3:

            explanation = (
                "The model predicts the Pancreatic Cancer "
                "category. This is not a confirmed diagnosis. "
                "Please seek professional medical assessment."
            )

        else:

            explanation = (
                "The prediction could not be interpreted. "
                "Please consult a healthcare professional."
            )

        # Map prediction to diagnosis
        diagnosis_map = {
            1: "No Cancer",
            2: "Non-Cancerous Pancreatitis",
            3: "Pancreatic Cancer"
        }

        prediction_result = diagnosis_map.get(
            predicted_class,
            "Unknown"
        )

        # Calculate training-set metrics
        # These are NOT independent test-set metrics.
        y_pred = clf.predict(X)

        accuracy = accuracy_score(y, y_pred)

        precision = precision_score(
            y,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y,
            y_pred,
            average="weighted",
            zero_division=0
        )

        return render_template(
            "result.html",
            show_default_menu=False,
            prediction=prediction_result,
            explanation=explanation,
            higher_values=higher_values,
            accuracy=f"Accuracy: {accuracy * 100:.2f}%",
            precision=f"Precision: {precision:.2f}",
            recall=f"Recall: {recall:.2f}",
            f1_score=f"F1-Score: {f1:.2f}",
        )

    return render_template(
        "prediction.html",
        show_default_menu=False
    )


# --------------------------------------------------
# 9. RECOMMENDATIONS PAGE
# --------------------------------------------------

@app.route("/prediction/recommendations")
def recommendations():

    return render_template(
        "recommendations.html",
        show_default_menu=False
    )


# --------------------------------------------------
# 10. ANALYSIS PAGE
# --------------------------------------------------

@app.route("/prediction/analyse")
def analyse():

    user_data = session.get("user_data")

    if not user_data:
        return redirect(url_for("prediction"))

    chart_img = create_chart(user_data)

    return render_template(
        "analyse.html",
        show_default_menu=False,
        chart_img=chart_img
    )


# --------------------------------------------------
# 11. RUN FLASK APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
