"""Reproducible measurements and explicit hypothetical opportunity models."""
import csv
import math
from pathlib import Path


def metrics_from_counts(tp, fp, fn, tn):
    values = [tp, fp, fn, tn]
    if any(type(v) is not int or v < 0 for v in values) or sum(values) == 0:
        raise ValueError('Counts must be nonnegative integers with a nonempty sample')
    return {'precision': tp/(tp+fp) if tp+fp else None,
            'recall': tp/(tp+fn) if tp+fn else None,
            'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'n': sum(values), 'flagged': tp+fp}


def load_saved_comparison(path):
    """Counts are authoritative; never select the most flattering saved metric."""
    with Path(path).open() as f:
        rows = list(csv.DictReader(f))
    result = []
    for row in rows:
        counts = {key: int(row[key]) for key in ('tp', 'fp', 'fn', 'tn')}
        result.append({**row, **counts, **metrics_from_counts(**counts),
                       'label_basis': 'keyword_proxy_not_clinician_adjudicated'})
    return result


def opportunity_funnel(admissions, candidate_share, missed_share, valid_share,
                       payment_change_share, realization_share, incremental_payment,
                       review_cost_per_candidate, implementation_cost=0):
    """Scenario only: no code automatically earns an incremental payment."""
    rates = [candidate_share, missed_share, valid_share, payment_change_share, realization_share]
    costs = [admissions, incremental_payment, review_cost_per_candidate, implementation_cost]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in rates):
        raise ValueError('Shares must be finite and between zero and one')
    if any(not math.isfinite(v) or v < 0 for v in costs):
        raise ValueError('Volume and costs must be finite and nonnegative')
    candidates = admissions*candidate_share
    changes = candidates*missed_share*valid_share*payment_change_share*realization_share
    gross = changes*incremental_payment
    review = candidates*review_cost_per_candidate
    return {'candidates': candidates, 'realized_changes': changes,
            'gross': gross, 'review_cost': review,
            'net': gross-review-implementation_cost, 'basis': 'hypothetical_scenario'}


def claim_payment_delta(before_payment=None, after_payment=None, validated=False):
    """Unvalidated claims have unknown value, not an automatic per-code payout."""
    if not validated or before_payment is None or after_payment is None:
        return None
    if any(not math.isfinite(v) or v < 0 for v in [before_payment, after_payment]):
        raise ValueError('Validated payment amounts must be finite and nonnegative')
    return after_payment-before_payment
