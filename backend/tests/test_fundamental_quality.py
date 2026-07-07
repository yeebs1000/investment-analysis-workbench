"""Correctness checks for the fundamental quality overlay. Run directly:

    python -m tests.test_fundamental_quality
"""
from __future__ import annotations

from app.analytics import fundamental_quality as fq


def test_high_quality_business_scores_high():
    """Strong ROE/ROIC/margins, growing, low debt -> High quality."""
    great = {
        "roe_pct": 35.0, "roic_pct": 28.0, "net_margin_pct": 25.0,
        "gross_margin_pct": 60.0, "revenue_growth_yoy_pct": 20.0,
        "debt_to_equity": 0.2, "current_ratio": 2.5,
    }
    r = fq.score_quality(great)
    assert r is not None
    assert r["score_0_100"] >= fq.STRONG_QUALITY, r
    assert r["label"] == "High quality"


def test_weak_business_scores_low():
    """Poor returns, shrinking, over-levered -> Weak."""
    poor = {
        "roe_pct": 2.0, "roic_pct": 1.0, "net_margin_pct": 1.0,
        "gross_margin_pct": 12.0, "revenue_growth_yoy_pct": -8.0,
        "debt_to_equity": 3.5, "current_ratio": 0.7,
    }
    r = fq.score_quality(poor)
    assert r is not None
    assert r["score_0_100"] < fq.WEAK_QUALITY, r
    assert r["label"] == "Weak"
    assert any("⚠" in x for x in r["reasons"]), "red flags should be surfaced"


def test_ordering_high_beats_low():
    great = {"roe_pct": 30, "roic_pct": 20, "net_margin_pct": 22, "gross_margin_pct": 55,
             "revenue_growth_yoy_pct": 18, "debt_to_equity": 0.3, "current_ratio": 2.2}
    poor = {"roe_pct": 4, "roic_pct": 2, "net_margin_pct": 3, "gross_margin_pct": 18,
            "revenue_growth_yoy_pct": -5, "debt_to_equity": 3.0, "current_ratio": 0.8}
    assert fq.score_quality(great)["score_0_100"] > fq.score_quality(poor)["score_0_100"]


def test_thin_coverage_returns_none():
    """Below the minimum metric count -> None (honest 'can't say'), not a guess."""
    assert fq.score_quality({"roe_pct": 20.0}) is None
    assert fq.score_quality(None) is None
    assert fq.score_quality({}) is None


def test_partial_coverage_still_scores_and_reports_missing():
    """3+ metrics present -> scored on what's available; missing listed."""
    partial = {"roe_pct": 22.0, "net_margin_pct": 15.0, "current_ratio": 1.8}
    r = fq.score_quality(partial)
    assert r is not None
    assert "3/" in r["coverage"]
    assert "roic_pct" in r["missing"]
    assert "debt_to_equity" in r["missing"]


def test_debt_lower_is_better():
    """The debt/equity rubric must reward LOWER leverage."""
    low = {"roe_pct": 15, "net_margin_pct": 10, "current_ratio": 1.5, "debt_to_equity": 0.2}
    high = {"roe_pct": 15, "net_margin_pct": 10, "current_ratio": 1.5, "debt_to_equity": 3.0}
    assert fq.score_quality(low)["score_0_100"] > fq.score_quality(high)["score_0_100"]


def test_size_growth_tilt_underweights_mega_overweights_small_grower():
    """Core of the user thesis: a mega-cap gets a strong underweight, a small-cap
    high-grower on a roll gets a strong overweight, and the sizing multiplier
    tracks the tilt direction."""
    mega = fq.size_growth_tilt({"market_cap_musd": 3_000_000, "revenue_growth_yoy_pct": 8})
    small = fq.size_growth_tilt({"market_cap_musd": 1_500, "revenue_growth_yoy_pct": 45}, rel_strength_pct=25)

    assert mega["tilt"] <= -0.5 and mega["label"] == "Strong underweight"
    assert small["tilt"] >= 0.5 and small["label"] == "Strong overweight"
    assert mega["sizing_multiplier"] < 1.0 < small["sizing_multiplier"]  # under- vs over-weight sizing
    assert small["tilt"] > mega["tilt"]


def test_size_growth_tilt_needs_market_cap():
    """No market cap -> None (honest degrade, never a guessed size)."""
    assert fq.size_growth_tilt({"revenue_growth_yoy_pct": 30}) is None
    assert fq.size_growth_tilt(None) is None
    # a mid-cap with steady growth sits ~neutral (no strong steer either way)
    mid = fq.size_growth_tilt({"market_cap_musd": 20_000, "revenue_growth_yoy_pct": 10})
    assert mid["label"] == "Neutral"


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} fundamental-quality tests passed.")


if __name__ == "__main__":
    main()
