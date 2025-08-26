"""
Part A: HIGGS SVM pipeline as a script.

Usage:
  python src/partA_higgs_svm.py \
    --data data/HIGGS.csv \
    --sample-frac 0.01 \
    --k-best 10 \
    --random-state 42 \
    --no-eda \
    --no-shap

Outputs:
  - outputs/part_A/figures/*.png
  - outputs/part_A/metrics/*.json
  - outputs/part_A/artifacts/*
"""

import os
import json
import time
import argparse
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, List
import shap
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

from joblib import Parallel, delayed

# Optional plotting imports guarded for headless environments
warnings.filterwarnings("ignore")
try:
    import matplotlib
    matplotlib.use("Agg")  # for headless savefig
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTS = True
except Exception:
    HAS_PLOTS = False

# Optional SHAP (can be slow on large data). Guarded by CLI flag.
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


# -----------------------
# Config dataclass
# -----------------------
@dataclass
class Config:
    dataset_path: str = "data/HIGGS.csv"
    sample_frac: float = 0.01
    k_best: int = 10
    random_state: int = 42
    test_size: float = 0.2
    do_eda: bool = True
    do_shap: bool = False
    out_dir: str = "outputs/part_A"
    fig_dir: str = "outputs/part_A/figures"
    metrics_dir: str = "outputs/part_A/metrics"
    artifacts_dir: str = "outputs/part_A/artifacts"


# -----------------------
# Utilities
# -----------------------
def ensure_dirs(cfg: Config):
    for d in [cfg.out_dir, cfg.fig_dir, cfg.metrics_dir, cfg.artifacts_dir]:
        os.makedirs(d, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def timed(fn):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = fn(*args, **kwargs)
        return result, time.time() - t0
    return wrapper


def safe_plot(save_path: str):
    if HAS_PLOTS:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


# -----------------------
# Data handling
# -----------------------
def load_higgs(file_path: str, sample_frac: float, random_state: int) -> pd.DataFrame:
    # HIGGS CSV: first col target (1=signal, 0=background), next 28 features
    data_org = pd.read_csv(file_path, header=None)
    if 0 < sample_frac < 1.0:
        data = data_org.sample(frac=sample_frac, random_state=random_state)
    else:
        data = data_org
    return data


# -----------------------
# Pipeline class
# -----------------------
class HIGGSDatasetAnalysis:
    def __init__(self, data: pd.DataFrame, cfg: Config):
        self.cfg = cfg
        self.data = data
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.X_train_scaled = None
        self.X_test_scaled = None

    def preprocess(self):
        self.y = self.data.iloc[:, 0]
        self.X = self.data.iloc[:, 1:]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.cfg.test_size, random_state=self.cfg.random_state
        )
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        return self

    def eda(self):
        if not HAS_PLOTS or not self.cfg.do_eda:
            return
        # Basic stats
        y_counts = self.y.value_counts(normalize=True)
        with open(os.path.join(self.cfg.metrics_dir, "target_distribution.txt"), "w") as f:
            f.write(str(y_counts))

        # Boxplot of features
        plt.figure(figsize=(16, 8))
        self.X.boxplot()
        plt.title("Feature Distributions")
        plt.xticks(rotation=90)
        safe_plot(os.path.join(self.cfg.fig_dir, "feature_distributions.png"))

        # Correlation heatmap
        plt.figure(figsize=(14, 10))
        sns.heatmap(self.X.corr(), cmap="coolwarm", center=0)
        plt.title("Feature Correlation Heatmap")
        safe_plot(os.path.join(self.cfg.fig_dir, "correlation_heatmap.png"))

    def feature_engineering(self):
        # Example: add 2-degree poly on first few features (lightweight)
        n_poly_base = min(5, self.X_train.shape[1])
        poly = PolynomialFeatures(degree=2, include_bias=False)
        Xtr_poly = poly.fit_transform(self.X_train.iloc[:, :n_poly_base])
        Xte_poly = poly.transform(self.X_test.iloc[:, :n_poly_base])
        # Keep only the interaction and higher-order terms beyond original
        Xtr_aug = np.hstack([self.X_train.values, Xtr_poly[:, n_poly_base:]])
        Xte_aug = np.hstack([self.X_test.values, Xte_poly[:, n_poly_base:]])
        # Refit scaler on augmented features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(Xtr_aug)
        self.X_test_scaled = self.scaler.transform(Xte_aug)
        return self

    def feature_selection(self, k_best: int) -> Tuple[np.ndarray, List[str]]:
        selector = SelectKBest(score_func=f_classif, k=k_best)
        X_new = selector.fit_transform(self.X_train_scaled, self.y_train)
        selected_indices = selector.get_support(indices=True)
        # Create dummy column names for augmented space
        feature_names = [f"f{i}" for i in range(self.X_train_scaled.shape[1])]
        selected_features = [feature_names[i] for i in selected_indices]
        # Persist selection
        save_json(
            {"k_best": k_best, "selected_indices": selected_indices.tolist(), "selected_features": selected_features},
            os.path.join(self.cfg.artifacts_dir, "select_kbest.json"),
        )
        return X_new, selected_features

    @timed
    def train_linear_svm_sgd(self):
        linear_svm = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            max_iter=1000,
            tol=1e-3,
            random_state=self.cfg.random_state,
        )
        cv_scores = cross_val_score(linear_svm, self.X_train_scaled, self.y_train, cv=5)
        linear_svm.fit(self.X_train_scaled, self.y_train)
        y_pred = linear_svm.predict(self.X_test_scaled)
        metrics = {
            "cv_scores": cv_scores.tolist(),
            "cv_mean": float(np.mean(cv_scores)),
            "accuracy": float(accuracy_score(self.y_test, y_pred)),
            "precision": float(precision_score(self.y_test, y_pred)),
            "recall": float(recall_score(self.y_test, y_pred)),
            "f1": float(f1_score(self.y_test, y_pred)),
        }
        return linear_svm, metrics

    def _evaluate_kernel_model(self, svm: SVC) -> Dict[str, Any]:
        start_time = time.time()
        svm.fit(self.X_train_scaled, self.y_train)
        training_time = time.time() - start_time

        start_pred = time.time()
        y_pred = svm.predict(self.X_test_scaled)
        pred_time = time.time() - start_pred

        # Probabilities if available -> AUC
        auc = None
        if hasattr(svm, "predict_proba"):
            y_proba = svm.predict_proba(self.X_test_scaled)[:, 1]
            try:
                auc = float(roc_auc_score(self.y_test, y_proba))
            except Exception:
                auc = None
        else:
            # fallback via decision_function if needed
            if hasattr(svm, "decision_function"):
                try:
                    scores = svm.decision_function(self.X_test_scaled)
                    y_prob = 1 / (1 + np.exp(-scores))
                    auc = float(roc_auc_score(self.y_test, y_prob))
                except Exception:
                    auc = None

        return {
            "Accuracy": float(accuracy_score(self.y_test, y_pred)),
            "Precision": float(precision_score(self.y_test, y_pred)),
            "Recall": float(recall_score(self.y_test, y_pred)),
            "F1 Score": float(f1_score(self.y_test, y_pred)),
            "AUC": auc,
            "Training Time": training_time,
            "Prediction Time": pred_time,
        }

    def kernel_comparison(self) -> Dict[str, Dict[str, Any]]:
        configs = []
        # Polynomial degrees
        for d in [2, 3, 4]:
            configs.append(("Polynomial", {"kernel": "poly", "degree": d, "coef0": 1.0, "C": 1.0, "gamma": "scale"}))
        # RBF variants
        for g in ["scale", "auto", 0.1, 1.0]:
            configs.append(("RBF", {"kernel": "rbf", "gamma": g, "C": 1.0}))
        # Sigmoid
        configs.append(("Sigmoid", {"kernel": "sigmoid", "gamma": "scale", "coef0": 1.0, "C": 1.0}))

        def run_one(name, params):
            svm = SVC(random_state=self.cfg.random_state, cache_size=500, class_weight="balanced", probability=False, **params)
            result = self._evaluate_kernel_model(svm)
            label = name
            if name == "Polynomial":
                label = f"Polynomial (degree={params.get('degree')})"
            elif name == "RBF":
                label = f"RBF (gamma={params.get('gamma')})"
            elif name == "Sigmoid":
                label = f"Sigmoid (gamma={params.get('gamma')})"
            return label, result

        results = dict(Parallel(n_jobs=-1, prefer="threads", verbose=0)(
            delayed(run_one)(name, params) for name, params in configs
        ))
        # Persist to JSON and CSV
        results_df = pd.DataFrame.from_dict(results, orient="index")
        results_df.sort_values("Accuracy", ascending=False).to_csv(
            os.path.join(self.cfg.artifacts_dir, "kernel_comparison.csv")
        )
        save_json(results, os.path.join(self.cfg.metrics_dir, "kernel_comparison.json"))
        return results

    @timed
    def randomized_tuning_sgd(self):
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, random_state=self.cfg.random_state))
        ])
        param_dist = {
            "svm__alpha": [1e-4, 1e-3, 1e-2],
            "svm__eta0": [0.1, 0.01],
            "svm__learning_rate": ["constant", "optimal"]
        }
        rs = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=6,
            cv=3,
            scoring="accuracy",
            n_jobs=-1,
            random_state=self.cfg.random_state
        )
        rs.fit(self.X_train_scaled, self.y_train)
        y_pred = rs.predict(self.X_test_scaled)
        metrics = {
            "best_params": rs.best_params_,
            "best_cv_score": float(rs.best_score_),
            "accuracy": float(accuracy_score(self.y_test, y_pred)),
            "precision": float(precision_score(self.y_test, y_pred)),
            "recall": float(recall_score(self.y_test, y_pred)),
            "f1": float(f1_score(self.y_test, y_pred)),
        }
        save_json(metrics, os.path.join(self.cfg.metrics_dir, "randomized_tuning_sgd.json"))
        return rs, metrics

    def hyperparameter_sensitivity_rbf(self) -> pd.DataFrame:
        C_range = [0.1, 1, 10, 100]
        gamma_range = [0.1, 1, 10, 100]

        def eval_pair(C, gamma):
            svm = SVC(kernel="rbf", C=C, gamma=gamma, random_state=self.cfg.random_state)
            svm.fit(self.X_train_scaled, self.y_train)
            y_pred = svm.predict(self.X_test_scaled)
            return float(accuracy_score(self.y_test, y_pred))

        combos = [(C, g) for C in C_range for g in gamma_range]
        scores = Parallel(n_jobs=-1, verbose=0)(
            delayed(eval_pair)(C, g) for C, g in combos
        )
        heat = np.array(scores).reshape(len(C_range), len(gamma_range))
        df = pd.DataFrame(heat, index=[f"C={c}" for c in C_range], columns=[f"gamma={g}" for g in gamma_range])
        df.to_csv(os.path.join(self.cfg.artifacts_dir, "rbf_sensitivity.csv"))

        if HAS_PLOTS:
            plt.figure(figsize=(8, 6))
            sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".3f")
            plt.title("RBF Sensitivity (Accuracy)")
            safe_plot(os.path.join(self.cfg.fig_dir, "rbf_sensitivity_heatmap.png"))
        return df

    def shap_analysis(self, model):
        if not HAS_SHAP or not self.cfg.do_shap:
            return None
        # Wrap for probability
        class ProbWrapper:
            def __init__(self, model_inner):
                self.model = model_inner
            def predict_proba(self, X):
                if hasattr(self.model, "predict_proba"):
                    return self.model.predict_proba(X)
                if hasattr(self.model, "decision_function"):
                    s = self.model.decision_function(X)
                    p = 1 / (1 + np.exp(-s))
                    return np.vstack((1 - p, p)).T
                # fallback to predictions
                p = self.model.predict(X)
                return np.vstack((1 - p, p)).T

        wrapped = ProbWrapper(model)
        Xtr_arr = self.X_train_scaled
        Xte_arr = self.X_test_scaled
        # Use kmeans summary to speed up
        background = shap.kmeans(Xtr_arr, 100)
        explainer = shap.KernelExplainer(lambda x: wrapped.predict_proba(x)[:, 1], background)
        sample = min(100, len(Xte_arr))
        shap_values = explainer.shap_values(Xte_arr[:sample])

        # Bar summary
        if HAS_PLOTS:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, Xte_arr[:sample], show=False, plot_type="bar")
            safe_plot(os.path.join(self.cfg.fig_dir, "shap_importance_bar.png"))
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, Xte_arr[:sample], show=False)
            safe_plot(os.path.join(self.cfg.fig_dir, "shap_summary.png"))

        # Save a small npy
        np.save(os.path.join(self.cfg.artifacts_dir, "shap_values.npy"), shap_values)
        return "saved"


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Part A - HIGGS SVM Pipeline")
    parser.add_argument("--data", type=str, default="data/HIGGS.csv", help="Path to HIGGS CSV")
    parser.add_argument("--sample-frac", type=float, default=0.01, help="Sampling fraction (0-1)")
    parser.add_argument("--k-best", type=int, default=10, help="SelectKBest top-k features")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--no-eda", action="store_true", help="Disable EDA plots/exports")
    parser.add_argument("--no-shap", action="store_true", help="Disable SHAP analysis")
    args = parser.parse_args()

    cfg = Config(
        dataset_path=args.data,
        sample_frac=args.sample_frac,
        k_best=args.k_best,
        random_state=args.random_state,
        do_eda=not args.no_eda,
        do_shap=not args.no_shap,
    )

    ensure_dirs(cfg)

    # Config export
    save_json(asdict(cfg), os.path.join(cfg.artifacts_dir, "config.json"))

    # Load
    print(f"[INFO] Loading dataset: {cfg.dataset_path} (sample_frac={cfg.sample_frac})")
    data = load_higgs(cfg.dataset_path, cfg.sample_frac, cfg.random_state)
    with open(os.path.join(cfg.metrics_dir, "dataset_shape.txt"), "w") as f:
        f.write(str(data.shape))

    # Pipeline
    analysis = HIGGSDatasetAnalysis(data, cfg).preprocess()
    if cfg.do_eda:
        print("[INFO] Running EDA...")
        analysis.eda()

    print("[INFO] Feature engineering...")
    analysis.feature_engineering()

    print(f"[INFO] SelectKBest (k={cfg.k_best})...")
    X_sel, sel_features = analysis.feature_selection(cfg.k_best)
    # Note: downstream models currently use full scaled features; selection saved for review.

    print("[INFO] Linear SVM (SGD) ...")
    (linear_model, linear_metrics), linear_time = analysis.train_linear_svm_sgd()
    linear_metrics["train_total_seconds"] = linear_time
    save_json(linear_metrics, os.path.join(cfg.metrics_dir, "linear_svm_sgd.json"))
    print(f"  Linear SGD - Acc: {linear_metrics['accuracy']:.3f}, F1: {linear_metrics['f1']:.3f}")

    print("[INFO] Kernel SVM comparison...")
    kernel_results = analysis.kernel_comparison()

    print("[INFO] Randomized tuning (SGD)...")
    (best_rs, rs_metrics), rs_time = analysis.randomized_tuning_sgd()
    rs_metrics["search_total_seconds"] = rs_time
    save_json(rs_metrics, os.path.join(cfg.metrics_dir, "randomized_tuning_summary.json"))
    print(f"  Best SGD params: {rs_metrics['best_params']}, Acc: {rs_metrics['accuracy']:.3f}")

    print("[INFO] Hyperparameter sensitivity (RBF)...")
    sensitivity_df = analysis.hyperparameter_sensitivity_rbf()

    if cfg.do_shap:
        print("[INFO] SHAP analysis on best RS model...")
        try:
            analysis.shap_analysis(best_rs.best_estimator_)
        except Exception as e:
            with open(os.path.join(cfg.metrics_dir, "shap_error.txt"), "w") as f:
                f.write(str(e))

    # Collate summary
    summary = {
        "linear_sgd": linear_metrics,
        "randomized_tuning": rs_metrics,
        "kernel_comparison_top": pd.DataFrame.from_dict(kernel_results, orient="index")
            .sort_values("Accuracy", ascending=False).head(5).to_dict(orient="index"),
    }
    save_json(summary, os.path.join(cfg.metrics_dir, "summary.json"))

    print("[INFO] Done. All figures, models, metrics, and artifacts saved under outputs/.")


if __name__ == "__main__":
    main()
