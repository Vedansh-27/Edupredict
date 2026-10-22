"""
EduPredict Cloud — API Performance Testing Script
==================================================
Measures response latency, success rate, and percentile timings for
the local or deployed prediction API.

Usage (local):
    python scripts/performance_test.py --url http://localhost:8000 --n 100

Usage (deployed — after getting your Render URL):
    python scripts/performance_test.py --url https://your-app.onrender.com --n 50

WARNING: Do not run aggressive load tests against the free Render tier.
         Limit to ≤50 requests with delays when testing the live service.
"""

import argparse
import json
import statistics
import sys
import time
from datetime import datetime

# Use only stdlib so no extra install is needed
try:
    import urllib.request
    import urllib.error
except ImportError:
    print("urllib not available — this should not happen.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Sample payload
# ---------------------------------------------------------------------------
PAYLOAD = {
    "G1": 12, "G2": 13, "studytime": 2, "failures": 0,
    "absences": 4, "sex": "M", "age": 17, "Medu": 2, "Fedu": 2,
    "famrel": 4, "freetime": 3, "goout": 3, "Dalc": 1, "Walc": 2, "health": 4,
}


def send_request(url: str) -> tuple[bool, float, int]:
    """
    Sends one POST /predict request.
    Returns (success: bool, latency_ms: float, status_code: int).
    """
    body = json.dumps(PAYLOAD).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return True, elapsed_ms, resp.status
    except urllib.error.HTTPError as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return False, elapsed_ms, e.code
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return False, elapsed_ms, 0


def percentile(data: list[float], p: float) -> float:
    """Simple percentile calculation."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def run_test(base_url: str, n: int, delay_s: float, is_remote: bool):
    predict_url = f"{base_url.rstrip('/')}/predict"
    health_url  = f"{base_url.rstrip('/')}/health"

    print("\n" + "=" * 60)
    print("EduPredict Cloud - Performance Test")
    print("=" * 60)
    print(f"Timestamp  : {datetime.now().isoformat()}")
    print(f"Target URL : {predict_url}")
    print(f"Requests   : {n}")
    print(f"Delay      : {delay_s}s between requests")
    print(f"Environment: {'Remote (Render)' if is_remote else 'Local'}")
    print("=" * 60 + "\n")

    # Check health first
    try:
        req = urllib.request.Request(health_url, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            health_data = json.loads(resp.read())
            model_ok = health_data.get("model_loaded", False)
            print(f"Health check: status={health_data.get('status')} model_loaded={model_ok}")
    except Exception as e:
        print(f"Health check FAILED: {e}")
        print("Aborting test — ensure the server is running.")
        sys.exit(1)

    print()

    latencies = []
    successes = 0
    failures  = 0

    for i in range(1, n + 1):
        ok, lat, code = send_request(predict_url)
        latencies.append(lat)

        if ok:
            successes += 1
            status_str = f"OK  ({lat:7.1f} ms)"
        else:
            failures += 1
            status_str = f"FAIL HTTP {code} ({lat:7.1f} ms)"

        sys.stdout.write(f"\r  Request {i:>4}/{n}  {status_str}")
        sys.stdout.flush()

        if delay_s > 0 and i < n:
            time.sleep(delay_s)

    print("\n")

    # Results
    avg_lat  = statistics.mean(latencies)
    p50      = percentile(latencies, 50)
    p95      = percentile(latencies, 95)
    min_lat  = min(latencies)
    max_lat  = max(latencies)
    err_rate = (failures / n) * 100

    print("-" * 40)
    print(f"  Total requests     : {n}")
    print(f"  Successful         : {successes}")
    print(f"  Failed             : {failures}")
    print(f"  Error rate         : {err_rate:.1f}%")
    print("-" * 40)
    print("  Latency (ms):")
    print(f"    Min              : {min_lat:.1f}")
    print(f"    Average          : {avg_lat:.1f}")
    print(f"    p50 (median)     : {p50:.1f}")
    print(f"    p95              : {p95:.1f}")
    print(f"    Max              : {max_lat:.1f}")
    print("-" * 40)
    print("\nTest complete.\n")

    if is_remote and n > 50:
        print("[WARNING] You ran more than 50 requests against the live service.")
        print("          The free Render plan has limited resources. Be considerate.")


def main():
    parser = argparse.ArgumentParser(description="EduPredict Cloud performance test.")
    parser.add_argument("--url", default="http://localhost:8000",
                        help="Base URL of the application (default: http://localhost:8000)")
    parser.add_argument("--n", type=int, default=50,
                        help="Number of requests to send (default: 50)")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Seconds between requests (default: 0). Use >0 for remote.")
    args = parser.parse_args()

    is_remote = "localhost" not in args.url and "127.0.0.1" not in args.url
    if is_remote and args.delay == 0.0:
        args.delay = 0.1   # auto-add small delay for remote targets
        print(f"[INFO] Remote target detected. Auto-setting delay to {args.delay}s.")

    run_test(args.url, args.n, args.delay, is_remote)


if __name__ == "__main__":
    main()
