import pandas as pd
import geopandas as gpd
import numpy as np
from libpysal.weights import Queen
import logging
import os
import sys
import time
import warnings
import multiprocessing
multiprocessing.freeze_support()


def _configure_pytensor_for_frozen_runtime():
    """
    Configure PyTensor before importing pymc.
    This avoids C-linker path issues inside PyInstaller one-file temp dirs.
    """
    if not getattr(sys, "frozen", False):
        return

    cache_root = os.path.join(
        os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
        "HjBM",
        "pytensor_cache",
    )
    os.makedirs(cache_root, exist_ok=True)

    desired = {
        "base_compiledir": cache_root,
        "compiledir_format": "compiledir_%(platform)s-%(python_version)s-%(python_bitwidth)s",
        "linker": "py",
        "cxx": "",
        "mode": "FAST_COMPILE",
    }

    existing = os.environ.get("PYTENSOR_FLAGS", "")
    parsed = {}
    if existing.strip():
        for item in existing.split(","):
            if "=" in item:
                k, v = item.split("=", 1)
                parsed[k.strip()] = v.strip()
            elif item.strip():
                parsed[item.strip()] = ""

    parsed.update(desired)
    os.environ["PYTENSOR_FLAGS"] = ",".join(
        f"{k}={v}" if v != "" else k for k, v in parsed.items()
    )


_configure_pytensor_for_frozen_runtime()

import pymc as pm
import arviz as az

def _setup_pipeline_logging(verbose_log_path, stream=None):
    """
    Verbose file log (DEBUG) plus optional stream (INFO) for terminal/GUI.
    pymc / pytensor DEBUG go to the same file only (no propagate).
    """
    log_dir = os.path.dirname(os.path.abspath(verbose_log_path))
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("pipeline")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    fmt_file = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fmt_stream = logging.Formatter("%(message)s")

    fh = logging.FileHandler(verbose_log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt_file)
    logger.addHandler(fh)

    if stream is not None:
        sh = logging.StreamHandler(stream)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt_stream)
        logger.addHandler(sh)

    for name in ("pymc", "pytensor"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.setLevel(logging.DEBUG)
        lg.addHandler(fh)
        lg.propagate = False

    return logger

def _build_icar_components(neighbors, n):
    """
    Build the node1/node2 edge index arrays needed for the ICAR prior.
    Returns (node1, node2) as arrays of ints, one entry per undirected edge.
    """
    node1, node2 = [], []
    for i, neighs in neighbors.items():
        for j in neighs:
            if i < j:
                node1.append(i)
                node2.append(j)
    return np.array(node1, dtype=int), np.array(node2, dtype=int)

def run_pipeline(params):
    """
    Run the full public health Bayesian pipeline.
    params keys:
      data_path, geo_path, id_col, outcome_col, exposure_col,
      min_val, max_val, exclude_cols (list, optional),
      artifact_stem, verbose_log_path, report_path (preferred), or legacy
      output_path + log_path for paths derived from splitext(output_path).
    """
    data_path    = params["data_path"]
    geo_path     = params["geo_path"]
    id_col       = params["id_col"]
    outcome_col  = params["outcome_col"]
    exposure_col = params["exposure_col"]
    min_val      = params["min_val"]
    max_val      = params["max_val"]
    tune_steps   = int(params.get("tune_steps", 2000))
    draw_steps   = int(params.get("draw_steps", 2000))
    target_accept = float(params.get("target_accept", 0.95))
    exclude_cols = params.get("exclude_cols", [])

    artifact_stem = params.get("artifact_stem")
    verbose_log_path = params.get("verbose_log_path") or params.get("log_path")
    report_path = params.get("report_path")
    if not artifact_stem or not verbose_log_path:
        legacy_out = params.get("output_path", "model_output.out")
        root, _ = os.path.splitext(legacy_out)
        if not artifact_stem:
            artifact_stem = root
        if not verbose_log_path:
            verbose_log_path = params.get("log_path", root + ".log")
    if not report_path:
        report_path = artifact_stem + "_report.out"

    logger = _setup_pipeline_logging(verbose_log_path, stream=sys.stdout)

    def log(msg):
        logger.info(msg)

    log("[1] Loading data...")
    if data_path.endswith(".xlsx"):
        df = pd.read_excel(data_path)
    elif data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
    else:
        raise ValueError("Unsupported file type. Use utf-8 CSV or XLSX.")

    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(r"[^\w\s]", "_", regex=True)
        .str.replace(r"\s+", "_", regex=True)
    )

    # Convert to parquet alongside source
    parquet_path = os.path.splitext(data_path)[0] + ".parquet"
    df.to_parquet(parquet_path, index=False)
    log(f"[2] Converted to parquet: {parquet_path}")

    # Drop excluded columns (keep required ones regardless)
    required = {id_col, outcome_col, exposure_col}
    to_drop = [c for c in exclude_cols if c in df.columns and c not in required]
    if to_drop:
        df = df.drop(columns=to_drop)
        log(f"    Excluded columns: {to_drop}")

    # Cleaning
    log("[3] Cleaning data...")
    df[outcome_col]  = pd.to_numeric(df[outcome_col],  errors="coerce")
    df[exposure_col] = pd.to_numeric(df[exposure_col], errors="coerce")
    df = df.dropna(subset=[id_col, outcome_col, exposure_col])

    # Assertions
    assert (df[outcome_col] >= min_val).all(), "Outcome below minimum"
    assert (df[outcome_col] <= max_val).all(), "Outcome above maximum"
    assert (df[exposure_col] > 0).all(), "Exposure must be > 0"

    df[id_col] = df[id_col].astype(str).str.zfill(5)

    # Spatial join
    log("[4] Loading spatial data...")
    gdf = gpd.read_file(geo_path)
    gdf.columns = gdf.columns.str.lower()
    if id_col not in gdf.columns:
        raise ValueError(f"'{id_col}' not found in spatial file columns: {list(gdf.columns)}")
    gdf[id_col] = gdf[id_col].astype(str).str.zfill(5)
    gdf = gdf.merge(df, on=id_col)
    log(f"    Merged {len(gdf)} records.")

    # Build adjacency
    log("[5] Building adjacency matrix...")
    w = Queen.from_dataframe(gdf)
    neighbors = w.neighbors
    n = len(gdf)

    # Number of neighbors per region (for the ICAR scaling)
    D = np.array([len(neighbors[i]) for i in range(n)], dtype=float)

    # Edge indices for ICAR prior
    node1, node2 = _build_icar_components(neighbors, n)
    n_edges = len(node1)
    log(f"    {n_edges} adjacency edges found.")

    # Bayesian model — BYM2 specification
    # phi  = structured spatial effect (ICAR)
    # theta = unstructured iid effect
    # Combined as: b = (sqrt(rho/s) * phi + sqrt(1-rho) * theta) * sigma
    # where rho mixes spatial vs non-spatial variance and s is a scaling
    # factor derived from the graph structure.
    log("[6] Running Bayesian model (this may take several minutes)...")
    y = gdf[outcome_col].values.astype(float)
    E = gdf[exposure_col].values.astype(float)
    log_E = np.log(E)

    # Scaling factor for the ICAR component (approximation via graph Laplacian)
    # This puts phi on unit variance scale so rho is interpretable.
    from scipy.sparse import diags as sp_diags
    from scipy.sparse.linalg import spsolve
    D_mat  = sp_diags(D)
    W_mat  = np.zeros((n, n))
    for i, neighs in neighbors.items():
        for j in neighs:
            W_mat[i, j] = 1
    Q = D_mat - W_mat  # graph Laplacian (precision matrix of ICAR)
    # Geometric mean of marginal variances of the ICAR (Riebler et al. 2016)
    Q_dense = Q.toarray() if hasattr(Q, "toarray") else Q
    # Add small jitter for numerical stability before pseudo-inverse
    Q_jitter = Q_dense + np.eye(n) * 1e-6
    Q_inv_diag = np.diag(np.linalg.pinv(Q_jitter))
    scaling_factor = float(np.exp(np.mean(np.log(Q_inv_diag))))
    log(f"    ICAR scaling factor: {scaling_factor:.4f}")

    # Hide noisy low-level numerical warnings from PyTensor internals in GUI output.
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        module=r"pytensor\.tensor\.elemwise",
    )

    # Progress callback for single-chain run.
    total_iterations = tune_steps + draw_steps
    samples_done = [0]
    last_emit = [0.0]

    def sampling_callback(trace, draw):
        samples_done[0] += 1
        now = time.time()
        done = samples_done[0] >= total_iterations
        if (not done) and (now - last_emit[0] < 1.5):
            return
        last_emit[0] = now
        pct = int((samples_done[0] / max(1, total_iterations)) * 100)
        pct = max(0, min(100, pct))
        bar_width = 40
        filled = int((pct / 100) * bar_width)
        bar = "#" * filled + "-" * (bar_width - filled)
        logger.info(
            f"    [SAMPLING][Chain 1] [{bar}] {pct}% | "
            f"Draw {samples_done[0]}/{total_iterations}"
        )

    with pm.Model() as model:
        # Intercept
        alpha = pm.Normal("alpha", mu=0, sigma=1)

        # BYM2 hyperpriors
        sigma = pm.HalfNormal("sigma", sigma=1)       # overall SD of random effects
        rho   = pm.Beta("rho", alpha=0.5, beta=0.5)   # proportion of spatial variance

        # Unstructured (iid) component
        theta = pm.Normal("theta", mu=0, sigma=1, shape=n)

        # Structured (ICAR) component via soft sum-to-zero constraint
        # phi_raw ~ ICAR(neighbors): penalty on differences between neighbors
        phi_raw = pm.Flat("phi_raw", shape=n)
        # ICAR soft constraint: penalize squared differences across edges
        phi_diff = phi_raw[node1] - phi_raw[node2]
        pm.Potential("icar_penalty", -0.5 * pm.math.sum(phi_diff ** 2))
        # Soft sum-to-zero
        pm.Potential("phi_sum_to_zero", -0.5 * (pm.math.sum(phi_raw) ** 2) / n)

        # Scale phi to unit variance using scaling factor
        phi = phi_raw / pm.math.sqrt(scaling_factor)

        # BYM2 combined random effect
        b = sigma * (pm.math.sqrt(rho) * phi + pm.math.sqrt(1 - rho) * theta)

        # Linear predictor with log offset
        mu = pm.math.exp(alpha + b + log_E)

        # Likelihood
        obs = pm.Poisson("obs", mu=mu, observed=y)

        trace = pm.sample(
            draw_steps,
            tune=tune_steps,
            cores=1,
            chains=1,
            target_accept=target_accept,
            progressbar=False,
            callback=sampling_callback
        )

    # Diagnostics
    log("[7] Running diagnostics...")
    summary = az.summary(trace)
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", None,
        "display.max_colwidth", None,
    ):
        summary_text = summary.to_string()
    if (summary["r_hat"] > 1.01).any():
        log("WARNING: Convergence issues detected (r_hat > 1.01)")
        logger.warning("Convergence issues: r_hat > 1.01 for some parameters.")

    # Output (CSVs in output_dir root; report in output_dir; verbose already in logs/)
    posterior_csv_path = artifact_stem + ".csv"
    summary_csv_path = artifact_stem + "_summary.csv"

    log(f"[8] Saving output to {posterior_csv_path} ...")
    # Reconstruct posterior mean of the Poisson rate for each region
    alpha_s = trace.posterior["alpha"].values        # (chain, draw)
    theta_s = trace.posterior["theta"].values        # (chain, draw, n)
    phi_s   = trace.posterior["phi_raw"].values      # (chain, draw, n)
    sigma_s = trace.posterior["sigma"].values        # (chain, draw)
    rho_s   = trace.posterior["rho"].values          # (chain, draw)

    phi_scaled = phi_s / np.sqrt(scaling_factor)
    b_s = (
        sigma_s[..., np.newaxis]
        * (
            np.sqrt(rho_s[..., np.newaxis])   * phi_scaled
            + np.sqrt(1 - rho_s[..., np.newaxis]) * theta_s
        )
    )
    rate = np.exp(alpha_s[..., np.newaxis] + b_s + log_E)
    posterior_mean = rate.mean(axis=(0, 1))

    gdf["posterior_mean"] = posterior_mean
    result_df = gdf.drop(columns="geometry")
    result_df.to_csv(posterior_csv_path, index=False, encoding="utf-8")
    summary.to_csv(summary_csv_path, index=True, encoding="utf-8")

    log("=== HBM Pipeline Summary ===")
    log(f"Input:   {data_path}")
    log(f"Spatial: {geo_path}")
    log(f"N regions: {n}")
    log(f"N edges:   {n_edges}")
    log(f"Scaling factor: {scaling_factor:.4f}")
    log("")
    log(summary_text)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== HBM Pipeline Summary ===\n")
        f.write(f"Input:   {data_path}\n")
        f.write(f"Spatial: {geo_path}\n")
        f.write(f"N regions: {n}\n")
        f.write(f"N edges:   {n_edges}\n")
        f.write(f"Scaling factor: {scaling_factor:.4f}\n\n")
        f.write(summary_text)
        f.write("\n")

    log(f"    Report: {report_path}")
    log(f"    Verbose log: {verbose_log_path}")
    log(f"    Posterior-by-region CSV: {posterior_csv_path}")
    log(f"    Diagnostic summary CSV: {summary_csv_path}")
    log(f"[DONE] Pipeline complete.")
