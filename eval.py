"""
eval.py — DeepEval evaluation harness for MindAI.

Measures the pipeline's answer quality and grounding using LLM-as-judge
metrics against a small hand-authored gold dataset:
  - FaithfulnessMetric: does the answer avoid claims unsupported by the
    retrieved context? (the core anti-hallucination check)
  - AnswerRelevancyMetric: does the answer actually address the question?
  - ContextualRecallMetric: did retrieval surface the context that the
    expected answer actually needed?

Architectural notes:
  - The gold dataset is intentionally small (5 cases) and mock — it targets
    a fictional but structurally realistic repo shape (an `auth` module,
    a `database` module, etc.) so this file runs standalone without
    requiring a live ingested repo. Swap `retrieval_context` /
    `actual_output` generation for calls into `graph.answer_question(...)`
    against a real ingested collection to eval the live pipeline end-to-end
    (see `build_test_cases_from_live_pipeline` below).
  - DeepEval's metrics are judged by an LLM (GPT-4o by default here) rather
    than exact-match, because "did this answer hallucinate" and "is this
    contextually relevant" are not string-comparable properties.
"""

from __future__ import annotations

import os

from deepeval import assert_test, evaluate
from deepeval.metrics import AnswerRelevancyMetric, ContextualRecallMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

JUDGE_MODEL = os.environ.get("MINDAI_JUDGE_MODEL", "gpt-4o")

# Thresholds are deliberately strict (0.7) since this system's core promise
# is "verifiable, no hallucination" — a lenient bar would undercut the eval's
# purpose.
FAITHFULNESS_THRESHOLD = 0.75
ANSWER_RELEVANCY_THRESHOLD = 0.7
CONTEXTUAL_RECALL_THRESHOLD = 0.7


# --------------------------------------------------------------------------- #
# Mock gold dataset
# --------------------------------------------------------------------------- #
#
# Each case models a realistic question/answer/retrieval-context triple for
# a fictional repo with an `auth` module, a `database` module, a `utils`
# module, and an API layer. `retrieval_context` simulates what the hybrid
# retrieval + rerank + graph-expansion pipeline would have surfaced;
# `actual_output` simulates what the generation node produced given that
# context (with citations, matching graph.py's contract).

GOLD_DATASET: list[dict] = [
    {
        "input": "How does the login function validate a user's password?",
        "actual_output": (
            "The `login` function in `src/auth/service.py:42-58` calls "
            "`verify_password`, defined at `src/auth/hashing.py:12-20`, which "
            "compares the bcrypt hash of the submitted password against the "
            "stored hash using `bcrypt.checkpw`. If the check fails, "
            "`login` raises an `AuthenticationError` (`src/auth/service.py:55-57`)."
        ),
        "expected_output": (
            "login() calls verify_password(), which uses bcrypt.checkpw to compare "
            "the submitted password's hash against the stored hash, raising "
            "AuthenticationError on mismatch."
        ),
        "retrieval_context": [
            "src/auth/service.py:42-58\ndef login(username, password):\n    user = get_user(username)\n"
            "    if not verify_password(password, user.password_hash):\n"
            "        raise AuthenticationError('Invalid credentials')\n    return create_session(user)",
            "src/auth/hashing.py:12-20\ndef verify_password(plain_password, hashed):\n"
            "    return bcrypt.checkpw(plain_password.encode(), hashed.encode())",
        ],
    },
    {
        "input": "What database connection pooling strategy does this project use?",
        "actual_output": (
            "The project uses SQLAlchemy's `QueuePool`, configured in "
            "`src/database/engine.py:8-15`, with `pool_size=10` and "
            "`max_overflow=20`. The engine is created once at module import "
            "time via `create_engine` (`src/database/engine.py:8-10`)."
        ),
        "expected_output": (
            "SQLAlchemy QueuePool is used, configured with pool_size=10 and "
            "max_overflow=20 in the engine setup module."
        ),
        "retrieval_context": [
            "src/database/engine.py:8-15\nengine = create_engine(\n    DATABASE_URL,\n"
            "    poolclass=QueuePool,\n    pool_size=10,\n    max_overflow=20,\n)",
        ],
    },
    {
        "input": "What functions call the `send_email` utility, and under what conditions?",
        "actual_output": (
            "`send_email` (`src/utils/mailer.py:5-18`) is called from two places: "
            "`register_user` (`src/auth/service.py:70-85`), which sends a welcome "
            "email after successful registration, and `request_password_reset` "
            "(`src/auth/service.py:90-102`), which sends a reset link. Both call "
            "sites wrap the call in a try/except to avoid failing the parent "
            "operation if email delivery fails."
        ),
        "expected_output": (
            "send_email is called by register_user (welcome email after signup) and "
            "request_password_reset (password reset link), both wrapping the call "
            "defensively so email failures don't break the parent flow."
        ),
        "retrieval_context": [
            "src/utils/mailer.py:5-18\ndef send_email(to, subject, body):\n"
            "    try:\n        smtp_client.send(to, subject, body)\n"
            "    except SMTPException:\n        logger.error('email failed')",
            "src/auth/service.py:70-85\ndef register_user(data):\n    user = create_user(data)\n"
            "    try:\n        send_email(user.email, 'Welcome', WELCOME_BODY)\n"
            "    except Exception:\n        pass\n    return user",
            "src/auth/service.py:90-102\ndef request_password_reset(email):\n"
            "    token = generate_reset_token(email)\n"
            "    try:\n        send_email(email, 'Reset your password', RESET_BODY.format(token))\n"
            "    except Exception:\n        pass\n    return token",
        ],
    },
    {
        "input": "Does this codebase implement rate limiting on the public API?",
        "actual_output": "I don't have enough information in the retrieved context to answer this confidently.",
        "expected_output": (
            "The retrieved context does not contain any rate-limiting middleware, "
            "decorator, or configuration, so the system should refuse rather than "
            "guess whether rate limiting exists."
        ),
        "retrieval_context": [
            "src/api/routes.py:1-20\nfrom fastapi import APIRouter\nrouter = APIRouter()\n\n"
            "@router.get('/users/{user_id}')\ndef get_user_endpoint(user_id: int):\n"
            "    return get_user(user_id)",
        ],
    },
    {
        "input": "How is the JWT access token's expiration time configured?",
        "actual_output": (
            "The access token expiration is set via `ACCESS_TOKEN_EXPIRE_MINUTES` "
            "(`src/auth/config.py:3`), currently `30`, and consumed in "
            "`create_access_token` (`src/auth/tokens.py:15-24`), which passes "
            "`timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)` as the `exp` claim "
            "when encoding the JWT with `jwt.encode`."
        ),
        "expected_output": (
            "ACCESS_TOKEN_EXPIRE_MINUTES (set to 30) is used by create_access_token "
            "to set the JWT's exp claim via timedelta, passed to jwt.encode."
        ),
        "retrieval_context": [
            "src/auth/config.py:3\nACCESS_TOKEN_EXPIRE_MINUTES = 30",
            "src/auth/tokens.py:15-24\ndef create_access_token(data):\n    expire = datetime.utcnow() + "
            "timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)\n    to_encode = data.copy()\n"
            "    to_encode.update({'exp': expire})\n    return jwt.encode(to_encode, SECRET_KEY, algorithm='HS256')",
        ],
    },
]


def build_test_cases_from_gold_dataset() -> list[LLMTestCase]:
    """Materialize the mock gold dataset into DeepEval LLMTestCase objects."""
    return [
        LLMTestCase(
            input=case["input"],
            actual_output=case["actual_output"],
            expected_output=case["expected_output"],
            retrieval_context=case["retrieval_context"],
        )
        for case in GOLD_DATASET
    ]


def build_test_cases_from_live_pipeline(
    github_url: str,
    questions: list[dict],
) -> list[LLMTestCase]:
    """
    Optional: run the *actual* ingestion + graph pipeline against a real
    repo and build test cases from its live output, instead of the mock
    dataset above. `questions` is a list of
    {"input": ..., "expected_output": ...} dicts (no retrieval_context —
    that's supplied by the live retriever).

    This is the path to use once you have a real repo indexed and want to
    validate the deployed pipeline rather than the mock fixtures.
    """
    from graph import answer_question
    from ingestion import ingest_repository

    _result, client, call_graph = ingest_repository(github_url)

    test_cases = []
    for q in questions:
        final_state = answer_question(q["input"], client, call_graph)
        reranked = final_state.get("reranked_chunks", [])
        expanded = final_state.get("expanded_chunks", [])
        context_strings = [
            f"{c['file_path']}:{c['start_line']}-{c['end_line']}\n{c['content']}"
            for c in (*reranked, *expanded)
        ]
        test_cases.append(
            LLMTestCase(
                input=q["input"],
                actual_output=final_state["answer"],
                expected_output=q.get("expected_output", ""),
                retrieval_context=context_strings or ["<no context retrieved>"],
            )
        )
    return test_cases


# --------------------------------------------------------------------------- #
# Metric construction
# --------------------------------------------------------------------------- #

def build_metrics() -> list:
    """
    Construct the three required DeepEval metrics, all judged by
    `JUDGE_MODEL` (GPT-4o by default) for maximum eval reliability —
    separate from the (cheaper) model used for actual generation.
    """
    return [
        FaithfulnessMetric(
            threshold=FAITHFULNESS_THRESHOLD,
            model=JUDGE_MODEL,
            include_reason=True,
        ),
        AnswerRelevancyMetric(
            threshold=ANSWER_RELEVANCY_THRESHOLD,
            model=JUDGE_MODEL,
            include_reason=True,
        ),
        ContextualRecallMetric(
            threshold=CONTEXTUAL_RECALL_THRESHOLD,
            model=JUDGE_MODEL,
            include_reason=True,
        ),
    ]


# --------------------------------------------------------------------------- #
# Pytest-compatible test entrypoints (run via `deepeval test run eval.py`)
# --------------------------------------------------------------------------- #

import pytest  # noqa: E402  (kept near usage for readability)


@pytest.mark.parametrize("test_case", build_test_cases_from_gold_dataset())
def test_mindai_pipeline_quality(test_case: LLMTestCase) -> None:
    """
    Per-case assertion against all three metrics. Run with:
        deepeval test run eval.py
    """
    metrics = build_metrics()
    assert_test(test_case, metrics)


# --------------------------------------------------------------------------- #
# Standalone script entrypoint (prints a full report without pytest)
# --------------------------------------------------------------------------- #

def run_evaluation_report() -> None:
    test_cases = build_test_cases_from_gold_dataset()
    metrics = build_metrics()
    results = evaluate(test_cases=test_cases, metrics=metrics)

    print("\n" + "=" * 80)
    print("MindAI Evaluation Report")
    print("=" * 80)
    for test_result in results.test_results:
        print(f"\nQuestion: {test_result.input}")
        print(f"  Success: {test_result.success}")
        for metric_data in test_result.metrics_data:
            print(f"  - {metric_data.name}: score={metric_data.score:.3f} "
                  f"(threshold={metric_data.threshold}) success={metric_data.success}")
            if metric_data.reason:
                print(f"    reason: {metric_data.reason}")


if __name__ == "__main__":
    run_evaluation_report()
