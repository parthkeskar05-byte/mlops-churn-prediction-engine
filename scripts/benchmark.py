"""Latency benchmarking script for FastAPI endpoint."""

import requests
import time
import numpy as np
from typing import List, Dict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def benchmark_endpoint(
    url: str,
    payload: Dict,
    n_requests: int = 100
) -> Dict[str, float]:
    """
    Benchmark API endpoint latency.

    Args:
        url: API endpoint URL
        payload: Request payload
        n_requests: Number of requests to send

    Returns:
        Dictionary with latency statistics
    """
    logger.info(f"Starting benchmark: {n_requests} requests to {url}")

    latencies: List[float] = []
    errors = 0

    for i in range(n_requests):
        try:
            start_time = time.time()
            response = requests.post(url, json=payload, timeout=10)
            end_time = time.time()

            if response.status_code == 200:
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
            else:
                errors += 1
                logger.warning(f"Request {i+1} failed with status {response.status_code}")

        except Exception as e:
            errors += 1
            logger.error(f"Request {i+1} error: {e}")

    if not latencies:
        raise RuntimeError("All requests failed")

    # Calculate statistics
    stats = {
        'total_requests': n_requests,
        'successful_requests': len(latencies),
        'failed_requests': errors,
        'mean_latency_ms': np.mean(latencies),
        'median_latency_ms': np.median(latencies),
        'p50_latency_ms': np.percentile(latencies, 50),
        'p95_latency_ms': np.percentile(latencies, 95),
        'p99_latency_ms': np.percentile(latencies, 99),
        'min_latency_ms': np.min(latencies),
        'max_latency_ms': np.max(latencies),
        'std_latency_ms': np.std(latencies)
    }

    return stats


def main():
    """Run benchmark."""
    # API endpoint
    api_url = "http://localhost:8000/predict"

    # Sample payload
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 844.20
    }

    # Check if API is available
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            logger.error("API not available. Start the API server first.")
            return
    except Exception as e:
        logger.error(f"Cannot connect to API: {e}")
        logger.info("Start the API with: python -m src.api.app")
        return

    # Run benchmark
    try:
        stats = benchmark_endpoint(api_url, payload, n_requests=100)

        # Print results
        logger.info("=" * 60)
        logger.info("BENCHMARK RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total requests:        {stats['total_requests']}")
        logger.info(f"Successful requests:   {stats['successful_requests']}")
        logger.info(f"Failed requests:       {stats['failed_requests']}")
        logger.info("-" * 60)
        logger.info(f"Mean latency:          {stats['mean_latency_ms']:.2f} ms")
        logger.info(f"Median latency:        {stats['median_latency_ms']:.2f} ms")
        logger.info(f"P50 latency:           {stats['p50_latency_ms']:.2f} ms")
        logger.info(f"P95 latency:           {stats['p95_latency_ms']:.2f} ms")
        logger.info(f"P99 latency:           {stats['p99_latency_ms']:.2f} ms")
        logger.info(f"Min latency:           {stats['min_latency_ms']:.2f} ms")
        logger.info(f"Max latency:           {stats['max_latency_ms']:.2f} ms")
        logger.info(f"Std deviation:         {stats['std_latency_ms']:.2f} ms")
        logger.info("=" * 60)

        # Check target metrics
        if stats['p95_latency_ms'] < 10.0:
            logger.info("✅ Target achieved: P95 < 10ms")
        else:
            logger.warning(f"⚠️  P95 latency {stats['p95_latency_ms']:.2f}ms exceeds 10ms target")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")


if __name__ == "__main__":
    main()
