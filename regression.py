############################################################
# Regression Suite: Poisson, Negative Binomial,           #
# and Stepwise Backwards Negative Binomial                 #
# For use with HjBM GUI                                    #
# H. Jachec  |  2026                                       #
############################################################

import os
import sys
import logging
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ── Logger ────────────────────────────────────────────────────────────────────

def _setup_logger(log_path, stream=None):
    log_dir = os.path.dirname(os.path.abspath(log_path))
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("regression")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt_file = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fmt_stream = logging.Formatter("%(message)s")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt_file)
    logger.addHandler(fh)
    if stream is not None:
        sh = logging.StreamHandler(stream)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt_stream)
        logger.addHandler(sh)
    return logger


# ── Data loader ───────────────────────────────────────────────────────────────

def _load_and_prepare(params):
    """
    Load CSV/XLSX, clean column names, resolve dependent variable and predictors.
    Returns: (y, X_raw, dep_var, predictors, log_fn, output_path)
    """
    data_path    = params["data_path"]
    dep_var      = params.get("dep_var", "").strip()
    exclude_cols = params.get("exclude_cols", [])
    output_path  = params.get("output_path", "regression_output.out")
    log_path     = params.get("log_path", "regression.log")

    logger = _setup_logger(log_path, stream=sys.stdout)

    def log(msg):
        logger.info(msg)

    # Load file
    log(f"[1] Loading data: {data_path}")
    if data_path.endswith(".xlsx"):
        df = pd.read_excel(data_path)
    elif data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
    else:
        raise ValueError("Unsupported file type. Use CSV or XLSX.")

    # Standardise column names
    df.columns = (
        df.columns
          .str.strip()
          .str.replace(r"[^\w\s]", "_", regex=True)
          .str.replace(r"\s+", "_", regex=True)
          .str.lower()
    )

    # Normalise dependent variable name
    dep_var_norm = dep_var.lower().replace(" ", "_") if dep_var else ""
    if not dep_var_norm or dep_var_norm not in df.columns:
        raise ValueError(
            f"Dependent variable '{dep_var}' not found in columns.\n"
            f"Available columns: {list(df.columns)}"
        )

    log(f"    Dependent variable : {dep_var_norm}")

    # Build exclusion set (always excludes dep var)
    exclude_norm = set(
        c.strip().lower().replace(" ", "_")
        for c in exclude_cols if c.strip()
    )
    exclude_norm.add(dep_var_norm)

    predictors = [c for c in df.columns if c not in exclude_norm]
    log(f"    Predictors ({len(predictors)}): {predictors}")

    # Coerce to numeric, drop missing / infinite
    for col in [dep_var_norm] + predictors:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[[dep_var_norm] + predictors].replace([np.inf, -np.inf], np.nan).dropna()

    log(f"    Clean observations : {len(df)}")

    y     = df[dep_var_norm]
    X_raw = df[predictors]

    return y, X_raw, dep_var_norm, predictors, log, output_path


# ── VIF ───────────────────────────────────────────────────────────────────────

def _compute_vif(X):
    """Compute VIF for each column in X (should already include the constant)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vif = pd.DataFrame({
            "Variable": X.columns,
            "VIF": [
                variance_inflation_factor(X.values, i)
                for i in range(X.shape[1])
            ]
        })
    return vif


# ── Output writer ─────────────────────────────────────────────────────────────

def _write_output(output_path, model_name, summary_str, vif_df, extra_sections=None):
    """
    Write results to .out file.
    extra_sections: list of (title, content_str) tuples appended at the end.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{'='*70}\n")
        f.write(f"  {model_name}\n")
        f.write(f"{'='*70}\n\n")
        f.write(summary_str)
        f.write(f"\n\n{'─'*70}\n")
        f.write("  Variance Inflation Factors (VIF)\n")
        f.write(f"{'─'*70}\n")
        f.write(vif_df.to_string(index=False))
        if extra_sections:
            for title, content in extra_sections:
                f.write(f"\n\n{'─'*70}\n")
                f.write(f"  {title}\n")
                f.write(f"{'─'*70}\n")
                f.write(content)
        f.write("\n")


# ── Poisson ───────────────────────────────────────────────────────────────────

def run_poisson(params):
    y, X_raw, dep_var, predictors, log, output_path = _load_and_prepare(params)

    log("\n[2] Fitting Poisson regression...")
    X = sm.add_constant(X_raw)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.GLM(y, X, family=sm.families.Poisson()).fit()

    log("\n--- Poisson Regression Results ---")
    log(str(model.summary()))

    vif_df = _compute_vif(X)
    log("\n--- VIF ---")
    log(vif_df.to_string(index=False))

    _write_output(output_path, "Poisson Regression", str(model.summary()), vif_df)
    log(f"\n[DONE] Output saved to {output_path}")


# ── Negative Binomial ─────────────────────────────────────────────────────────

def run_negative_binomial(params):
    y, X_raw, dep_var, predictors, log, output_path = _load_and_prepare(params)

    log("\n[2] Fitting Negative Binomial regression...")
    X = sm.add_constant(X_raw)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit()

    log("\n--- Negative Binomial Regression Results ---")
    log(str(model.summary()))

    vif_df = _compute_vif(X)
    log("\n--- VIF ---")
    log(vif_df.to_string(index=False))

    _write_output(output_path, "Negative Binomial Regression", str(model.summary()), vif_df)
    log(f"\n[DONE] Output saved to {output_path}")


# ── Stepwise Backwards Negative Binomial ─────────────────────────────────────

def run_stepwise_nb(params, threshold=0.05):
    """
    Backwards stepwise NB regression using p-value elimination.
    At each step, removes the predictor with the highest p-value above `threshold`.
    Stops when all remaining predictors are significant (p <= threshold).
    Mirrors the original manual script behaviour.
    """
    y, X_raw, dep_var, predictors, log, output_path = _load_and_prepare(params)

    current_predictors = list(predictors)

    log(f"\n[2] Starting Stepwise Backwards NB (p-value threshold = {threshold})...")
    log(f"    Starting predictors: {len(current_predictors)}")

    # Fit and log the full model first for reference
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_full    = sm.add_constant(X_raw[current_predictors])
        full_model = sm.GLM(y, X_full, family=sm.families.NegativeBinomial()).fit()

    log("\n--- Full Model Summary ---")
    log(str(full_model.summary()))
    log("\n[3] Beginning elimination...")

    # Stepwise p-value elimination
    while len(current_predictors) > 1:
        X_trial = sm.add_constant(X_raw[current_predictors])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                m = sm.GLM(
                    y, X_trial, family=sm.families.NegativeBinomial()
                ).fit(disp=False)
            except Exception as e:
                log(f"    [WARNING] Model fitting failed: {e}. Stopping elimination.")
                break

        # Find the predictor with the highest p-value (ignoring the intercept)
        pvals      = m.pvalues.drop("const", errors="ignore")
        worst_pval = pvals.max()
        worst_var  = pvals.idxmax()

        if worst_pval <= threshold:
            log(f"    All remaining predictors significant (p ≤ {threshold}). Stopping.")
            break

        log(f"    Removing '{worst_var}' (p = {worst_pval:.4f})")
        current_predictors.remove(worst_var)

    # Final model
    log(f"\n[4] Final model — {len(current_predictors)} predictor(s) retained:")
    log(f"    {current_predictors}")

    X_final = sm.add_constant(X_raw[current_predictors])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_model = sm.GLM(
            y, X_final, family=sm.families.NegativeBinomial()
        ).fit()

    log("\n--- Stepwise NB Final Model Summary ---")
    log(str(final_model.summary()))

    vif_df = _compute_vif(X_final)
    log("\n--- VIF (Final Model) ---")
    log(vif_df.to_string(index=False))

    # Coefficient table
    coef_df = pd.DataFrame({
        "Coefficient": final_model.params,
        "Std Error":   final_model.bse,
        "z":           final_model.tvalues,
        "p-value":     final_model.pvalues,
    })

    _write_output(
        output_path,
        "Stepwise Backwards Negative Binomial — Final Model",
        str(final_model.summary()),
        vif_df,
        extra_sections=[
            ("Full Model Summary (Pre-Elimination)", str(full_model.summary())),
            ("Final Coefficient Table",              coef_df.to_string()),
        ]
    )
    log(f"\n[DONE] Output saved to {output_path}")