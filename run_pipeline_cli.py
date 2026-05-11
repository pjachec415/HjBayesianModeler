import os
import time
import traceback
import argparse
from pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="HjBM Pipeline CLI")

    parser.add_argument("--data_path", required=True, help="Data file path (CSV/XLSX)")
    parser.add_argument("--geo_path", required=True, help="Spatial file path (GeoJSON/SHP)")
    parser.add_argument("--id_col", required=True, help="ID column name")
    parser.add_argument("--outcome_col", required=True, help="Outcome column name")
    parser.add_argument("--exposure_col", required=True, help="Exposure column name")
    parser.add_argument("--min_val", required=True, type=float, help="Min outcome value")
    parser.add_argument("--max_val", required=True, type=float, help="Max outcome value")
    parser.add_argument("--output_dir", required=True, help="Output directory")

    args = parser.parse_args()

    # Resolve paths
    data_path = os.path.abspath(os.path.expanduser(args.data_path))
    geo_path = os.path.abspath(os.path.expanduser(args.geo_path))

    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.dirname(data_path)
    else:
        output_dir = os.path.abspath(os.path.expanduser(output_dir))

    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    epoch = int(time.time())
    stem = os.path.splitext(os.path.basename(data_path))[0]
    base = f"HBM_{stem}_{epoch}"
    artifact_stem = os.path.join(output_dir, base)
    verbose_log_path = os.path.join(logs_dir, base + ".out")
    err_path = os.path.join(logs_dir, base + ".err")
    report_path = artifact_stem + "_report.out"

    params = {
        "data_path": data_path,
        "geo_path": geo_path,
        "id_col": args.id_col,
        "outcome_col": args.outcome_col,
        "exposure_col": args.exposure_col,
        "min_val": args.min_val,
        "max_val": args.max_val,
        "artifact_stem": artifact_stem,
        "verbose_log_path": verbose_log_path,
        "report_path": report_path,
        "exclude_cols": [],
    }

    print("=" * 60)
    print(" HjBM Pipeline CLI")
    print("=" * 60)
    print(f"  Data:       {params['data_path']}")
    print(f"  Geo:        {params['geo_path']}")
    print(f"  ID col:     {params['id_col']}")
    print(f"  Outcome:    {params['outcome_col']}")
    print(f"  Exposure:   {params['exposure_col']}")
    print(f"  Range:      [{params['min_val']}, {params['max_val']}]")
    print(f"  CSV/report: {output_dir}")
    print(f"  Verbose:    {verbose_log_path}")
    print(f"  Errors:     {err_path}")
    print("=" * 60 + "\n")

    try:
        run_pipeline(params)
    except Exception:
        with open(err_path, "w", encoding="utf-8") as ef:
            ef.write(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
