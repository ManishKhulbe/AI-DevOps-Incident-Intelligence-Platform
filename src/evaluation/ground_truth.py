"""
Ground truth dataset for evaluation.

What is a ground truth dataset?
It is a set of (question, expected_answer, relevant_contexts) triples that
YOU write by hand. Ragas compares your system's actual output against these
to produce objective quality scores.

Why write it by hand?
Because an AI system cannot evaluate itself. You need a human-defined
reference to measure against. Think of it like an exam answer key — you
cannot ask the student to write the answer key.

These scenarios match the three log scenarios seeded by scripts/seed_sample_logs.py.
If you add more seed scenarios, add corresponding entries here.

Structure of each entry:
    question         — the plain-English question a user would ask
    ground_truth     — the correct answer a perfect system should give
    expected_contexts — key phrases that MUST appear in the retrieved chunks
                        for the retrieval to be considered accurate
"""

GROUND_TRUTH: list[dict] = [
    # ── Scenario 1: DB connection pool exhaustion ──────────────────────────
    {
        "question": "Why did the payment service fail at 14:28?",
        "ground_truth": (
            "The payment service failed at 14:28 due to database connection pool exhaustion. "
            "All 100 connections to payments-db were in use, causing new requests to timeout "
            "after 30 seconds. This triggered the circuit breaker to open at 14:28:09, "
            "returning 503 errors to all downstream callers until the DB reconnected at 14:28:16."
        ),
        "expected_contexts": [
            "pool exhausted",
            "Connection timeout after 30s",
            "Circuit breaker OPEN",
            "payments-db",
        ],
    },

    # ── Scenario 2: OOMKilled restart loop ─────────────────────────────────
    {
        "question": "Which service was OOMKilled and how many times did it restart?",
        "ground_truth": (
            "The recommendation-service was OOMKilled 3 times. "
            "Each restart was caused by memory usage exceeding the 4000MB container limit "
            "during model inference with batch_size=512. "
            "After the third kill the pod entered CrashLoopBackOff."
        ),
        "expected_contexts": [
            "OOMKilled",
            "recommendation-service",
            "4000MB",
            "CrashLoopBackOff",
            "restart_count",
        ],
    },

    # ── Scenario 3: Deployment causing elevated error rate ──────────────────
    {
        "question": "What caused the elevated error rate in checkout at 16:00?",
        "ground_truth": (
            "Deployment v2.3.1 of checkout-service introduced a NullPointerException "
            "in CartSerializer.toJson(). The error rate reached 78% within 6 seconds of "
            "the new version receiving traffic, triggering an automatic rollback to v2.3.0. "
            "The service recovered to a 0.3% error rate by 16:00:45."
        ),
        "expected_contexts": [
            "NullPointerException",
            "CartSerializer",
            "v2.3.1",
            "error rate",
            "rollback",
        ],
    },

    # ── Scenario 4: Cross-scenario — deployment correlation ─────────────────
    {
        "question": "Was there a deployment before the checkout errors?",
        "ground_truth": (
            "Yes. Deployment v2.3.1 (deploy_id=deploy-789, git_sha=f4a91bc) started at 16:00:00, "
            "and the first NullPointerException errors appeared at 16:00:22 — "
            "22 seconds after the new pod started receiving traffic."
        ),
        "expected_contexts": [
            "deploy_id=deploy-789",
            "v2.3.1",
            "16:00:00",
            "NullPointerException",
        ],
    },

    # ── Scenario 5: Negative test — out of scope question ───────────────────
    {
        "question": "What was the CPU usage of the payment service during the incident?",
        "ground_truth": (
            "The available logs do not contain CPU usage metrics for the payment service. "
            "The logs only record connection pool state, timeouts, and circuit breaker events."
        ),
        "expected_contexts": [
            "Connection timeout",
            "pool exhausted",
        ],
    },
]
