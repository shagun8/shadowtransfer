"""
Master runner for cross-location diagnostic analysis.

Usage:
    # Run all diagnostics
    python run_diagnostics.py --threads all

    # Run specific threads
    python run_diagnostics.py --threads 1 3

    # Run specific sub-diagnostics
    python run_diagnostics.py --diagnostics 1a 1b 3e

    # Run only for specific resolutions or cities
    python run_diagnostics.py --threads all --resolutions highres --cities chicago phoenix
"""
import os
import sys
import argparse
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CITIES, RESOLUTIONS, OUTPUT_BASE, output_dir
from utils import load_city_data, precompute_gt_cache

from thread1_entanglement import (
    run_all_thread1, diagnostic_1a, diagnostic_1b, diagnostic_1c)
from thread3_geometry import (
    run_all_thread3, diagnostic_3a, diagnostic_3b, diagnostic_3e)
from thread4_position import (
    run_all_thread4, diagnostic_4a, diagnostic_4b)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run cross-location diagnostics")
    parser.add_argument(
        "--threads", nargs="+", default=["all"],
        help="Which threads to run: 1, 3, 4, or 'all'")
    parser.add_argument(
        "--diagnostics", nargs="+", default=None,
        help="Specific diagnostics to run: 1a, 1b, 1c, 1d, 3a, 3b, 3e, 4a, 4b")
    parser.add_argument(
        "--resolutions", nargs="+", default=None,
        help="Resolutions to evaluate (default: all)")
    parser.add_argument(
        "--cities", nargs="+", default=None,
        help="Cities to evaluate (default: all)")
    return parser.parse_args()


def load_all_city_data(cities=None, resolutions=None):
    """
    Load GT data for all (city, resolution) pairs into a cache.
    Done once and shared across all diagnostics.
    """
    cities      = cities      or CITIES
    resolutions = resolutions or RESOLUTIONS

    cache        = {}
    total_images = 0

    print("\n" + "=" * 70)
    print("LOADING GROUND TRUTH DATA")
    print("=" * 70)

    for city in cities:
        for res in resolutions:
            t0   = time.time()
            data = load_city_data(city, res)
            elapsed = time.time() - t0

            if data is not None:
                n             = len(data["filenames"])
                total_images += n
                cache[(city, res)] = data
                print(f"  {city}/{res}: {n} images loaded ({elapsed:.1f}s)")
            else:
                print(f"  {city}/{res}: FAILED TO LOAD")

    print(f"\nTotal: {total_images} images across {len(cache)} (city, res) pairs")
    return cache


def precompute_all_gt(cache):
    """
    Pre-compute GT-derived info (instances, geometry, eval crops) once
    per (city, res).
    """
    print("\n" + "=" * 70)
    print("PRE-COMPUTING GT CACHE (instances, geometry, eval crops)")
    print("=" * 70)

    for key, data in cache.items():
        city, res = key
        t0 = time.time()
        data["gt_cache"] = precompute_gt_cache(data)
        n_instances = sum(len(img["instances"]) for img in data["gt_cache"])
        elapsed = time.time() - t0
        print(f"  {city}/{res}: {n_instances} instances pre-computed "
              f"({elapsed:.1f}s)")

    return cache


def main():
    args = parse_args()

    t_start = time.time()

    # Determine what to run
    if args.diagnostics:
        diags = set(args.diagnostics)
    elif "all" in args.threads:
        diags = {"1a", "1b", "1c", "1d", "3a", "3b", "3e", "4a", "4b"}
    else:
        diags = set()
        for t in args.threads:
            if t == "1":
                diags.update({"1a", "1b", "1c", "1d"})
            elif t == "3":
                diags.update({"3a", "3b", "3e"})
            elif t == "4":
                diags.update({"4a", "4b"})

    print(f"\nDiagnostics to run: {sorted(diags)}")

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # Load and pre-compute once
    cache = load_all_city_data(
        cities=args.cities,
        resolutions=args.resolutions,
    )

    if not cache:
        print("ERROR: No data loaded. Check paths in config.py")
        sys.exit(1)

    cache = precompute_all_gt(cache)

    # Run requested diagnostics
    results = {}

    # Thread 1
    if "1a" in diags:
        results["1a"] = diagnostic_1a(cache)
    if "1b" in diags:
        results["1b"] = diagnostic_1b(cache)
    if "1c" in diags:
        results["1c"] = diagnostic_1c(cache)
    if "1d" in diags:
        # 1d is standalone (reads pre-extracted features from disk, not cache)
        try:
            from thread1_1d import diagnostic_1d
            results["1d"] = diagnostic_1d()
        except ImportError:
            print("  1d skipped: thread1_1d.py not available or "
                  "sklearn not installed")
        except Exception as e:
            print(f"  1d skipped: {e}")

    # Thread 3
    if "3e" in diags:
        results["3e"] = diagnostic_3e(cache)
    if "3a" in diags:
        results["3a"] = diagnostic_3a(cache)
    if "3b" in diags:
        results["3b"] = diagnostic_3b(cache)

    # Thread 4
    if "4a" in diags:
        results["4a"] = diagnostic_4a(cache)
    if "4b" in diags:
        results["4b"] = diagnostic_4b(cache)

    # Summary
    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"ALL DIAGNOSTICS COMPLETE ({elapsed:.1f}s)")
    print("=" * 70)
    print(f"Results saved to: {OUTPUT_BASE}")
    print(f"Diagnostics run: {sorted(diags)}")

    meta = {
        "diagnostics_run":  sorted(list(diags)),
        "cities":           args.cities      or CITIES,
        "resolutions":      args.resolutions or RESOLUTIONS,
        "elapsed_seconds":  elapsed,
    }
    with open(os.path.join(OUTPUT_BASE, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()