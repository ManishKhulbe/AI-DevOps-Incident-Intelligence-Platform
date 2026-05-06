"""
Seed the system with realistic synthetic DevOps logs covering three incident scenarios.

Run after `docker compose up -d` and `pip install -r requirements.txt`:

    python scripts/seed_sample_logs.py

Why synthetic logs for learning?
Real production logs often contain PII (emails, IPs, tokens). Synthetic logs let
you practice retrieval and RCA without handling sensitive data. The scenarios here
are modelled on the most common real-world incident patterns.

Scenarios:
    1. Payment service DB connection pool exhaustion → cascade timeout
    2. Kubernetes OOMKilled pod → service restart loop
    3. Deployment causing elevated error rate in checkout service
"""

import asyncio
import sys
from pathlib import Path

# Allow running from the project root: python scripts/seed_sample_logs.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.models import RawLog
from src.ingestion.pipeline import ingest

# ---------------------------------------------------------------------------
# Scenario 1 — DB connection pool exhaustion in payment-service
# ---------------------------------------------------------------------------
PAYMENT_DB_LOGS = """\
2024-01-15T14:28:01 INFO  payment-service host=payment-pod-1 Starting request processing request_id=abc-001
2024-01-15T14:28:02 INFO  payment-service host=payment-pod-1 Acquiring DB connection from pool size=98/100
2024-01-15T14:28:03 WARN  payment-service host=payment-pod-1 Connection pool at 99% capacity size=99/100
2024-01-15T14:28:05 ERROR payment-service host=payment-pod-1 Failed to acquire DB connection: pool exhausted trace_id=abc-001
2024-01-15T14:28:05 ERROR payment-service host=payment-pod-1 Connection timeout after 30s - all 100 connections in use
2024-01-15T14:28:06 ERROR payment-service host=payment-pod-1 Request abc-001 failed: upstream database unavailable
2024-01-15T14:28:07 ERROR payment-service host=payment-pod-2 Connection timeout after 30s trace_id=abc-002
2024-01-15T14:28:08 ERROR payment-service host=payment-pod-2 Connection timeout after 30s trace_id=abc-003
2024-01-15T14:28:09 CRITICAL payment-service host=payment-pod-1 Circuit breaker OPEN: 5 consecutive failures on payments-db:5432
2024-01-15T14:28:10 ERROR  payment-service host=payment-pod-1 All downstream requests returning 503 Service Unavailable
2024-01-15T14:28:11 INFO   payment-service host=payment-pod-1 Attempting DB reconnect in 5s
2024-01-15T14:28:16 INFO   payment-service host=payment-pod-1 DB reconnect successful - closing idle connections
2024-01-15T14:28:17 INFO   payment-service host=payment-pod-1 Circuit breaker HALF-OPEN: testing with 1 request
2024-01-15T14:28:18 INFO   payment-service host=payment-pod-1 Circuit breaker CLOSED: payments-db healthy
"""

# ---------------------------------------------------------------------------
# Scenario 2 — OOMKilled pod restart loop in recommendation-service
# ---------------------------------------------------------------------------
OOMKILLED_LOGS = """\
2024-01-15T15:10:00 INFO  recommendation-service host=rec-pod-3 Starting model inference batch_size=512
2024-01-15T15:10:02 INFO  recommendation-service host=rec-pod-3 Loaded embedding model memory_mb=1800
2024-01-15T15:10:05 WARN  recommendation-service host=rec-pod-3 Memory usage at 85% mem_used=3400MB mem_limit=4000MB
2024-01-15T15:10:08 WARN  recommendation-service host=rec-pod-3 Memory usage at 92% mem_used=3680MB mem_limit=4000MB
2024-01-15T15:10:10 WARN  recommendation-service host=rec-pod-3 Memory usage at 97% mem_used=3880MB mem_limit=4000MB
2024-01-15T15:10:12 ERROR recommendation-service host=rec-pod-3 OOMKilled: container exceeded memory limit 4000MB
2024-01-15T15:10:12 INFO  kubernetes node=worker-2 Pod rec-pod-3 OOMKilled - restart_count=1
2024-01-15T15:10:45 INFO  recommendation-service host=rec-pod-3 Container restarted restart_count=1
2024-01-15T15:10:47 INFO  recommendation-service host=rec-pod-3 Starting model inference batch_size=512
2024-01-15T15:10:55 ERROR recommendation-service host=rec-pod-3 OOMKilled: container exceeded memory limit 4000MB
2024-01-15T15:10:55 INFO  kubernetes node=worker-2 Pod rec-pod-3 OOMKilled - restart_count=2
2024-01-15T15:11:40 INFO  recommendation-service host=rec-pod-3 Container restarted restart_count=2
2024-01-15T15:11:42 ERROR recommendation-service host=rec-pod-3 OOMKilled: container exceeded memory limit 4000MB
2024-01-15T15:11:42 WARN  kubernetes node=worker-2 Pod rec-pod-3 in CrashLoopBackOff restart_count=3
2024-01-15T15:12:00 INFO  kubernetes node=worker-2 Applying backoff delay=20s before next restart
"""

# ---------------------------------------------------------------------------
# Scenario 3 — Bad deployment causing elevated error rate in checkout
# ---------------------------------------------------------------------------
DEPLOYMENT_LOGS = """\
2024-01-15T16:00:00 INFO  checkout-service host=checkout-pod-1 Deployment started version=v2.3.1 git_sha=f4a91bc deploy_id=deploy-789
2024-01-15T16:00:05 INFO  checkout-service host=checkout-pod-1 Rolling update: replacing v2.3.0 with v2.3.1
2024-01-15T16:00:10 INFO  checkout-service host=checkout-pod-2 New pod healthy version=v2.3.1
2024-01-15T16:00:15 INFO  checkout-service host=checkout-pod-1 Old pod terminating version=v2.3.0
2024-01-15T16:00:20 INFO  checkout-service host=checkout-pod-2 Serving traffic version=v2.3.1 deploy_id=deploy-789
2024-01-15T16:00:22 ERROR checkout-service host=checkout-pod-2 NullPointerException in CartSerializer.toJson() version=v2.3.1
2024-01-15T16:00:23 ERROR checkout-service host=checkout-pod-2 NullPointerException in CartSerializer.toJson() version=v2.3.1
2024-01-15T16:00:24 ERROR checkout-service host=checkout-pod-2 NullPointerException in CartSerializer.toJson() version=v2.3.1
2024-01-15T16:00:25 WARN  checkout-service host=checkout-pod-2 Error rate 45% — threshold is 5% deploy_id=deploy-789
2024-01-15T16:00:26 CRITICAL checkout-service host=checkout-pod-2 Error rate 78% — triggering automatic rollback deploy_id=deploy-789
2024-01-15T16:00:27 INFO  checkout-service host=checkout-pod-2 Rollback initiated: v2.3.1 → v2.3.0 deploy_id=deploy-789
2024-01-15T16:00:35 INFO  checkout-service host=checkout-pod-1 Rollback pod healthy version=v2.3.0
2024-01-15T16:00:40 INFO  checkout-service host=checkout-pod-2 Traffic shifted back to v2.3.0
2024-01-15T16:00:45 INFO  checkout-service host=checkout-pod-2 Error rate 0.3% — back to normal version=v2.3.0
"""


async def seed() -> None:
    scenarios = [
        RawLog(
            source="text",
            service_name="payment-service",
            environment="prod",
            content=PAYMENT_DB_LOGS,
        ),
        RawLog(
            source="text",
            service_name="recommendation-service",
            environment="prod",
            content=OOMKILLED_LOGS,
        ),
        RawLog(
            source="text",
            service_name="checkout-service",
            environment="prod",
            content=DEPLOYMENT_LOGS,
        ),
    ]

    total_chunks = 0
    for raw_log in scenarios:
        result = await ingest(raw_log)
        total_chunks += result["chunks_stored"]
        print(f"  [{raw_log.service_name}] log_id={result['log_id']}  chunks={result['chunks_stored']}")

    print(f"\nDone. {total_chunks} chunks stored across {len(scenarios)} scenarios.")
    print("You can now query the system:")
    print('  "Why did the payment service fail at 14:28?"')
    print('  "Which pod was OOMKilled and how many times did it restart?"')
    print('  "What caused the elevated error rate in checkout at 16:00?"')


if __name__ == "__main__":
    asyncio.run(seed())
