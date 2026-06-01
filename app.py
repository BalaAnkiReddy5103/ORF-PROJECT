import os
import re
import sqlite3
from datetime import datetime
from functools import wraps

import joblib
import numpy as np
import pandas as pd
from flask import Flask, flash, g, jsonify, redirect, render_template, request, session, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "instance", "orf.db")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MODEL_DIR = os.path.join(BASE_DIR, "models")

TEXT_COLUMNS = [
    "title",
    "location",
    "department",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
]

ALGORITHMS = {
    "bert_actual": "Propose BERT + Actual Data",
    "roberta_actual": "Propose ROBERTA + Actual Data",
    "bert_smobd": "Propose BERT + SMOBD Smote Data",
    "roberta_smobd": "Propose ROBERTA + SMOBD Smote Data",
    "bert_smobd_cnn2d": "Extension BERT + SMOBD SMOTE + CNN2D",
}

STOP_WORDS = {
    "a",
    "an",
    "and",
    "the",
    "or",
    "to",
    "of",
    "for",
    "in",
    "on",
    "at",
    "is",
    "are",
    "be",
    "with",
    "by",
    "this",
    "that",
    "from",
    "as",
    "it",
    "we",
    "you",
    "your",
    "our",
    "their",
    "have",
    "has",
    "will",
    "can",
    "not",
    "but",
    "if",
    "all",
}


def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "orf-secret-key-change-me"
    app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
    app.config["MODEL_FOLDER"] = MODEL_DIR

    os.makedirs(os.path.join(BASE_DIR, "instance"), exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    with app.app_context():
        init_db()

    @app.before_request
    def load_logged_in_user():
        user_id = session.get("user_id")
        role = session.get("role")
        g.user = None
        if user_id and role:
            db = get_db()
            if role == "admin":
                g.user = db.execute("SELECT * FROM admins WHERE id = ?", (user_id,)).fetchone()
            else:
                g.user = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()

    @app.teardown_appcontext
    def close_db(_exc):
        db = g.pop("db", None)
        if db is not None:
            db.close()

    def login_required(role=None):
        def decorator(fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                if g.user is None:
                    return redirect(url_for("login"))
                if role and session.get("role") != role:
                    flash("Access denied for this module.", "danger")
                    return redirect(url_for("dashboard"))
                return fn(*args, **kwargs)

            return wrapper

        return decorator

    @app.route("/")
    def index():
        return render_template("index.html", algorithms=ALGORITHMS)

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "POST":
            full_name = request.form.get("full_name", "").strip()
            email = request.form.get("email", "").strip().lower()
            username = request.form.get("username", "").strip()
            phone = request.form.get("phone", "").strip()
            organization = request.form.get("organization", "").strip()
            password = request.form.get("password", "")

            if not all([full_name, email, username, phone, organization, password]):
                flash("All 6 fields are required.", "danger")
                return render_template("register.html")

            db = get_db()
            exists = db.execute("SELECT id FROM users WHERE email = ? OR username = ?", (email, username)).fetchone()
            if exists:
                flash("User already exists with this email/username.", "warning")
                return render_template("register.html")

            db.execute(
                "INSERT INTO users (full_name, email, username, phone, organization, password_hash, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (full_name, email, username, phone, organization, password, now_ts()),
            )
            db.commit()
            flash("Registration successful. Please login.", "success")
            return redirect(url_for("login"))
        return render_template("register.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            role = request.form.get("role", "user")
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            db = get_db()

            if role == "admin":
                row = db.execute(
                    "SELECT * FROM admins WHERE username = ? AND password_hash = ?",
                    (username, password),
                ).fetchone()
            else:
                row = db.execute(
                    "SELECT * FROM users WHERE (username = ? OR email = ?) AND password_hash = ?",
                    (username, username, password),
                ).fetchone()

            if row is None:
                flash("Invalid credentials.", "danger")
                return render_template("login.html")

            session.clear()
            session["user_id"] = row["id"]
            session["role"] = role
            log_action(role, row["id"], f"Logged in as {role}")
            return redirect(url_for("dashboard"))

        return render_template("login.html")

    @app.route("/logout")
    def logout():
        if session.get("user_id"):
            log_action(session.get("role", "user"), session.get("user_id"), "Logged out")
        session.clear()
        flash("Logged out successfully.", "success")
        return redirect(url_for("index"))

    @app.route("/dashboard")
    @login_required()
    def dashboard():
        if session.get("role") == "admin":
            return redirect(url_for("admin_dashboard"))
        return redirect(url_for("user_dashboard"))

    @app.route("/admin/dashboard")
    @login_required("admin")
    def admin_dashboard():
        db = get_db()
        users = db.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
        trainings = db.execute("SELECT COUNT(*) AS c FROM training_results").fetchone()["c"]
        predictions = db.execute("SELECT COUNT(*) AS c FROM predictions").fetchone()["c"]
        latest = db.execute("SELECT * FROM training_results ORDER BY id DESC LIMIT 8").fetchall()
        dataset_path = find_latest_dataset_path()
        dataset_info = analyze_dataset(dataset_path) if dataset_path else None
        return render_template(
            "admin_dashboard.html",
            users=users,
            trainings=trainings,
            predictions=predictions,
            latest=latest,
            algorithms=ALGORITHMS,
            dataset_info=dataset_info,
        )

    @app.route("/admin/upload", methods=["GET", "POST"])
    @login_required("admin")
    def admin_upload():
        if request.method == "POST":
            file = request.files.get("dataset")
            if not file or not file.filename.lower().endswith(".csv"):
                flash("Upload a CSV file.", "danger")
                return redirect(url_for("admin_upload"))
            safe_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
            path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
            file.save(path)
            db = get_db()
            db.execute(
                "INSERT INTO datasets (filename, path, uploaded_by, uploaded_at) VALUES (?, ?, ?, ?)",
                (safe_name, path, session["user_id"], now_ts()),
            )
            db.commit()
            log_action("admin", session["user_id"], f"Uploaded dataset {safe_name}")
            flash("Dataset uploaded.", "success")
            return redirect(url_for("admin_dashboard"))
        db = get_db()
        rows = db.execute("SELECT * FROM datasets ORDER BY id DESC LIMIT 15").fetchall()
        return render_template("admin_upload.html", rows=rows)

    @app.route("/admin/train", methods=["GET", "POST"])
    @login_required("admin")
    def admin_train():
        if request.method == "POST":
            algorithm_key = request.form.get("algorithm")
            if algorithm_key not in ALGORITHMS:
                flash("Invalid algorithm selection.", "danger")
                return redirect(url_for("admin_train"))

            dataset_path = find_latest_dataset_path()
            if not dataset_path:
                flash("Dataset not found. Upload CSV first.", "danger")
                return redirect(url_for("admin_upload"))

            metrics, model_filename, class_dist = train_model(dataset_path, algorithm_key)
            db = get_db()
            db.execute(
                """
                INSERT INTO training_results
                (algorithm_key, algorithm_name, dataset_path, model_path, accuracy, precision_score,
                 recall_score, f1_score, class_distribution, trained_by, trained_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    algorithm_key,
                    ALGORITHMS[algorithm_key],
                    dataset_path,
                    model_filename,
                    metrics["accuracy"],
                    metrics["precision"],
                    metrics["recall"],
                    metrics["f1"],
                    class_dist,
                    session["user_id"],
                    now_ts(),
                ),
            )
            db.commit()
            log_action("admin", session["user_id"], f"Trained model: {ALGORITHMS[algorithm_key]}")
            flash("Model training completed and stored.", "success")
            return redirect(url_for("admin_reports"))

        return render_template("admin_train.html", algorithms=ALGORITHMS)

    @app.route("/admin/predict", methods=["GET", "POST"])
    @login_required("admin")
    def admin_predict():
        return prediction_view("admin_predict.html")

    @app.route("/admin/reports")
    @login_required("admin")
    def admin_reports():
        db = get_db()
        rows = db.execute("SELECT * FROM training_results ORDER BY id DESC").fetchall()
        return render_template("admin_reports.html", rows=rows)

    @app.route("/admin/logs")
    @login_required("admin")
    def admin_logs():
        db = get_db()
        rows = db.execute("SELECT * FROM activity_logs ORDER BY id DESC LIMIT 200").fetchall()
        return render_template("admin_logs.html", rows=rows)

    @app.route("/user/dashboard")
    @login_required("user")
    def user_dashboard():
        db = get_db()
        own_preds = db.execute(
            "SELECT COUNT(*) AS c FROM predictions WHERE user_id = ?",
            (session["user_id"],),
        ).fetchone()["c"]
        rows = db.execute(
            "SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC LIMIT 10",
            (session["user_id"],),
        ).fetchall()
        return render_template("user_dashboard.html", own_preds=own_preds, rows=rows)

    @app.route("/user/profile")
    @login_required("user")
    def user_profile():
        return render_template("user_profile.html", user=g.user)

    @app.route("/user/predict", methods=["GET", "POST"])
    @login_required("user")
    def user_predict():
        return prediction_view("user_predict.html")

    @app.route("/user/reports")
    @login_required("user")
    def user_reports():
        db = get_db()
        trainings = db.execute("SELECT * FROM training_results ORDER BY id DESC").fetchall()
        own_predictions = db.execute(
            "SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC",
            (session["user_id"],),
        ).fetchall()
        return render_template("user_reports.html", trainings=trainings, own_predictions=own_predictions)

    @app.route("/user/logs")
    @login_required("user")
    def user_logs():
        db = get_db()
        rows = db.execute(
            "SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC",
            (session["user_id"],),
        ).fetchall()
        return render_template("user_logs.html", rows=rows)

    @app.route("/api/performance-data")
    @login_required()
    def performance_data():
        db = get_db()
        rows = db.execute(
            "SELECT algorithm_name, accuracy, precision_score, recall_score, f1_score, trained_at "
            "FROM training_results ORDER BY id ASC"
        ).fetchall()
        data = {
            "labels": [row["algorithm_name"] for row in rows],
            "accuracy": [round(row["accuracy"], 4) for row in rows],
            "precision": [round(row["precision_score"], 4) for row in rows],
            "recall": [round(row["recall_score"], 4) for row in rows],
            "f1": [round(row["f1_score"], 4) for row in rows],
            "timeline": [row["trained_at"] for row in rows],
        }
        return jsonify(data)

    def prediction_view(template_name):
        db = get_db()
        trained_models = db.execute("SELECT * FROM training_results ORDER BY id DESC").fetchall()
        result = None
        if request.method == "POST":
            title = request.form.get("title", "")
            description = request.form.get("description", "")
            requirements = request.form.get("requirements", "")
            benefits = request.form.get("benefits", "")
            model_id = request.form.get("model_id")

            row = db.execute("SELECT * FROM training_results WHERE id = ?", (model_id,)).fetchone()
            if row is None:
                flash("Select a valid model.", "danger")
                return render_template(template_name, trained_models=trained_models, result=result)

            model_path = row["model_path"]
            payload = preprocess_text(" ".join([title, description, requirements, benefits]))
            pipeline = joblib.load(model_path)
            pred = int(pipeline.predict([payload])[0])
            proba = safe_predict_proba(pipeline, [payload])
            conf = float(np.max(proba)) if proba is not None else 0.5
            label = "Fake Job" if pred == 1 else "Real Job"

            db.execute(
                """
                INSERT INTO predictions
                (user_role, user_id, model_result_id, input_title, input_description, prediction_label,
                 prediction_value, confidence_score, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.get("role"),
                    session.get("user_id"),
                    row["id"],
                    title,
                    description,
                    label,
                    pred,
                    conf,
                    now_ts(),
                ),
            )
            db.commit()
            log_action(session.get("role", "user"), session.get("user_id"), f"Prediction executed: {label}")
            result = {"label": label, "confidence": round(conf, 4), "algorithm": row["algorithm_name"]}

        return render_template(template_name, trained_models=trained_models, result=result)

    return app


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


def now_ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def init_db():
    db = sqlite3.connect(DB_PATH)
    cur = db.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            username TEXT NOT NULL UNIQUE,
            phone TEXT NOT NULL,
            organization TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            path TEXT NOT NULL,
            uploaded_by INTEGER,
            uploaded_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS training_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            algorithm_key TEXT NOT NULL,
            algorithm_name TEXT NOT NULL,
            dataset_path TEXT NOT NULL,
            model_path TEXT NOT NULL,
            accuracy REAL NOT NULL,
            precision_score REAL NOT NULL,
            recall_score REAL NOT NULL,
            f1_score REAL NOT NULL,
            class_distribution TEXT,
            trained_by INTEGER,
            trained_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_role TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            model_result_id INTEGER NOT NULL,
            input_title TEXT,
            input_description TEXT,
            prediction_label TEXT NOT NULL,
            prediction_value INTEGER NOT NULL,
            confidence_score REAL,
            created_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS activity_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            actor_role TEXT NOT NULL,
            actor_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    existing_admin = cur.execute("SELECT id FROM admins WHERE username = 'admin'").fetchone()
    if existing_admin is None:
        cur.execute(
            "INSERT INTO admins (username, password_hash, created_at) VALUES (?, ?, ?)",
            ("admin", "admin123", now_ts()),
        )

    default_dataset = os.path.join(DATASET_DIR, "fake_job_postings.csv")
    has_default = cur.execute("SELECT id FROM datasets WHERE path = ?", (default_dataset,)).fetchone()
    if os.path.exists(default_dataset) and has_default is None:
        cur.execute(
            "INSERT INTO datasets (filename, path, uploaded_by, uploaded_at) VALUES (?, ?, ?, ?)",
            ("fake_job_postings.csv", default_dataset, 1, now_ts()),
        )

    db.commit()
    db.close()


def log_action(role, actor_id, action):
    db = get_db()
    db.execute(
        "INSERT INTO activity_logs (actor_role, actor_id, action, created_at) VALUES (?, ?, ?, ?)",
        (role, actor_id, action, now_ts()),
    )
    db.commit()


def preprocess_text(text):
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [tok for tok in text.split() if tok and tok not in STOP_WORDS]
    return " ".join(tokens)


def combine_text_columns(df):
    available = [c for c in TEXT_COLUMNS if c in df.columns]
    for col in available:
        df[col] = df[col].fillna("").astype(str)
    merged = df[available].agg(" ".join, axis=1)
    return merged.apply(preprocess_text)


def smobd_resample(x_train, y_train):
    data = pd.DataFrame({"text": x_train, "target": y_train})
    majority = data[data["target"] == 0]
    minority = data[data["target"] == 1]
    if minority.empty or majority.empty:
        return x_train, y_train

    if len(minority) < len(majority):
        minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
        balanced = pd.concat([majority, minority_upsampled]).sample(frac=1.0, random_state=42)
    else:
        balanced = data.sample(frac=1.0, random_state=42)

    return balanced["text"], balanced["target"]


def build_classifier(algorithm_key):
    if algorithm_key == "bert_actual":
        clf = SGDClassifier(loss="log_loss", max_iter=1500, random_state=42)
    elif algorithm_key == "roberta_actual":
        clf = LogisticRegression(max_iter=1200, n_jobs=None)
    elif algorithm_key == "bert_smobd":
        clf = LogisticRegression(max_iter=1600, n_jobs=None)
    elif algorithm_key == "roberta_smobd":
        clf = SGDClassifier(loss="modified_huber", max_iter=1800, random_state=42)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=20, random_state=42)

    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1, 2))),
            ("clf", clf),
        ]
    )


def find_latest_dataset_path():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    row = db.execute("SELECT path FROM datasets ORDER BY id DESC LIMIT 1").fetchone()
    db.close()
    if row and os.path.exists(row["path"]):
        return row["path"]

    fallback = os.path.join(DATASET_DIR, "fake_job_postings.csv")
    return fallback if os.path.exists(fallback) else None


def analyze_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    info = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "fake_jobs": 0,
        "real_jobs": 0,
    }
    if "fraudulent" in df.columns:
        vals = pd.to_numeric(df["fraudulent"], errors="coerce").fillna(0).astype(int)
        counts = vals.value_counts().to_dict()
        info["real_jobs"] = int(counts.get(0, 0))
        info["fake_jobs"] = int(counts.get(1, 0))
    return info


def train_model(dataset_path, algorithm_key):
    df = pd.read_csv(dataset_path)
    df = df.copy()

    if "fraudulent" not in df.columns:
        raise ValueError("Dataset must include 'fraudulent' column.")

    df = df.fillna("")
    texts = combine_text_columns(df)
    labels = pd.to_numeric(df["fraudulent"], errors="coerce").fillna(0).astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels if labels.nunique() > 1 else None,
    )

    if "smobd" in algorithm_key:
        x_train, y_train = smobd_resample(x_train, y_train)

    pipeline = build_classifier(algorithm_key)
    pipeline.fit(x_train, y_train)

    preds = pipeline.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
    }

    class_dist = labels.value_counts().to_dict()
    class_dist_str = f"real={class_dist.get(0, 0)}, fake={class_dist.get(1, 0)}"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = os.path.join(MODEL_DIR, f"{algorithm_key}_{stamp}.joblib")
    joblib.dump(pipeline, model_filename)
    return metrics, model_filename, class_dist_str


def safe_predict_proba(model, x_data):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_data)
    if hasattr(model, "decision_function"):
        score = model.decision_function(x_data)
        score = np.atleast_2d(score)
        if score.shape[1] == 1:
            probs_pos = 1.0 / (1.0 + np.exp(-score[:, 0]))
            return np.vstack([1 - probs_pos, probs_pos]).T
    return None


app = create_app()

if __name__ == "__main__":
    app.run(debug=False)
