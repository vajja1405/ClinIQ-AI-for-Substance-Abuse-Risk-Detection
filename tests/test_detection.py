"""
Unit tests for ClinIQ detection logic.
These tests are self-contained — no database or API key required.
"""

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (mirrors logic from analysis/task1_signal_detection.py)
# ─────────────────────────────────────────────────────────────────────────────

OPIOID_KEYWORDS = [
    'opioid', 'opiate', 'heroin', 'fentanyl', 'morphine',
    'hydrocodone', 'oxycodone', 'suboxone', 'buprenorphine',
    'methadone', 'naloxone', 'naltrexone', 'tramadol', 'codeine',
]

ALCOHOL_KEYWORDS = [
    'alcohol dependence', 'alcoholism', 'alcohol abuse',
    'alcohol use disorder', 'drinking problem', 'alcoholic',
]

WITHDRAWAL_KEYWORDS = [
    'withdrawal', 'withdrawals', 'detox', 'detoxing',
    'cold turkey', 'physically dependent',
]

ALL_KEYWORDS = OPIOID_KEYWORDS + ALCOHOL_KEYWORDS + WITHDRAWAL_KEYWORDS


def rule_based_classify(text: str, keywords: list[str]) -> bool:
    """Classify a review as SUD-positive if any keyword matches."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors."""
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def embedding_classify(
    review_vec: np.ndarray,
    reference_vecs: np.ndarray,
    threshold: float = 0.32,
) -> bool:
    """Return True if any reference vector is within cosine threshold."""
    sims = [cosine_similarity(review_vec, ref) for ref in reference_vecs]
    return max(sims) >= threshold


def revenue_impact(missed_codes: int, low: float = 2750, high: float = 6000) -> tuple[float, float]:
    """Estimate annual revenue impact from missed SUD comorbidity codes."""
    return missed_codes * low, missed_codes * high


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based detection tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRuleBasedDetection:
    def test_opioid_positive(self):
        text = "I have been on opioid therapy for 3 years and developed dependence."
        assert rule_based_classify(text, ALL_KEYWORDS) is True

    def test_alcohol_positive(self):
        text = "Struggling with alcoholism and trying to get sober."
        assert rule_based_classify(text, ALL_KEYWORDS) is True

    def test_withdrawal_positive(self):
        text = "Going cold turkey was the worst experience of my life."
        assert rule_based_classify(text, ALL_KEYWORDS) is True

    def test_non_sud_negative(self):
        text = "This medication helped my blood pressure significantly."
        assert rule_based_classify(text, ALL_KEYWORDS) is False

    def test_case_insensitive(self):
        text = "Diagnosed with OPIOID USE DISORDER last year."
        assert rule_based_classify(text, ALL_KEYWORDS) is True

    def test_partial_word_match(self):
        # 'heroin' appears inside 'heroine' — test that we don't false-positive
        # on words that merely contain a keyword as a substring
        text = "She was the heroine of the story, full of courage."
        # Note: this is a known limitation of simple substring matching
        # The test documents this behavior explicitly
        result = rule_based_classify(text, ALL_KEYWORDS)
        # 'heroin' is a substring of 'heroine' — rule-based will flag this
        assert result is True  # documents known recall/precision trade-off

    def test_fentanyl_detected(self):
        text = "The fentanyl patch was not enough to manage my chronic pain."
        assert rule_based_classify(text, ALL_KEYWORDS) is True

    def test_empty_text(self):
        assert rule_based_classify("", ALL_KEYWORDS) is False


# ─────────────────────────────────────────────────────────────────────────────
# Embedding-based detection tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbeddingDetection:
    def test_identical_vectors_full_similarity(self):
        v = np.array([1.0, 0.5, -0.3, 0.8])
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors_zero_similarity(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors_negative_similarity(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_zero_vector_returns_zero(self):
        a = np.zeros(4)
        b = np.array([1.0, 0.5, -0.3, 0.8])
        assert cosine_similarity(a, b) == 0.0

    def test_classify_above_threshold(self):
        # Review vector very close to a reference
        ref = np.array([1.0, 0.0, 0.0, 0.0])
        review = np.array([0.99, 0.14, 0.0, 0.0])  # cosine ~0.99
        refs = np.array([ref])
        assert embedding_classify(review, refs, threshold=0.32) is True

    def test_classify_below_threshold(self):
        ref = np.array([1.0, 0.0, 0.0, 0.0])
        review = np.array([0.3, 0.95, 0.0, 0.0])  # low similarity
        refs = np.array([ref])
        assert embedding_classify(review, refs, threshold=0.32) is False

    def test_classify_matches_any_reference(self):
        # Only second reference is close
        refs = np.array([
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0]),
        ])
        review = np.array([0.0, 0.99, 0.0, 0.0])  # close to second ref
        assert embedding_classify(review, refs, threshold=0.32) is True

    def test_384_dimensional_vector(self):
        """Sanity-check that cosine works on actual embedding dimension."""
        a = np.random.default_rng(42).normal(size=384)
        assert -1.0 <= cosine_similarity(a, a) <= 1.0 + 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Revenue impact tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRevenueImpact:
    def test_single_missed_code(self):
        lo, hi = revenue_impact(1)
        assert lo == 2750
        assert hi == 6000

    def test_ten_thousand_admissions(self):
        lo, hi = revenue_impact(10_000)
        assert lo == 27_500_000
        assert hi == 60_000_000

    def test_zero_missed_codes(self):
        lo, hi = revenue_impact(0)
        assert lo == 0
        assert hi == 0

    def test_low_less_than_high(self):
        lo, hi = revenue_impact(42)
        assert lo < hi


# ─────────────────────────────────────────────────────────────────────────────
# ICD-10 code format tests
# ─────────────────────────────────────────────────────────────────────────────

class TestICD10Codes:
    """Validate that SUD ICD-10 codes follow the expected F1x.xxx pattern."""

    SUD_CODES = [
        "F11.20",  # Opioid dependence, uncomplicated (CC)
        "F11.23",  # Opioid dependence with withdrawal (MCC)
        "F10.230", # Alcohol dependence with withdrawal (MCC)
        "F10.20",  # Alcohol dependence, uncomplicated (CC)
        "F19.10",  # Other psychoactive substance abuse
    ]

    def test_all_codes_start_with_F(self):
        for code in self.SUD_CODES:
            assert code.startswith("F"), f"{code} should start with F"

    def test_all_codes_have_decimal(self):
        for code in self.SUD_CODES:
            assert "." in code, f"{code} missing decimal separator"

    def test_mcc_withdrawal_codes(self):
        mcc_codes = {"F11.23", "F10.230"}
        for code in mcc_codes:
            assert code in self.SUD_CODES

    def test_no_duplicate_codes(self):
        assert len(self.SUD_CODES) == len(set(self.SUD_CODES))
