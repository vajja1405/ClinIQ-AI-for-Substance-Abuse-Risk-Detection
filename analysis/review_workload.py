"""Planning estimates from measured sensitivity/specificity; not clinical validation."""
import math
from numbers import Real

def estimate_workload(tp, fp, fn, tn, volume, prevalence, minutes_per_review):
    vals = (tp, fp, fn, tn, volume, prevalence, minutes_per_review)
    if any(isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x) for x in vals):
        raise ValueError("Inputs must be finite numbers")
    if any(x < 0 for x in vals) or prevalence > 1:
        raise ValueError("Counts/time must be nonnegative; prevalence must be between zero and one")
    if tp + fn <= 0 or tn + fp <= 0:
        raise ValueError("Evaluation must include both positive and negative examples")
    sensitivity = tp / (tp + fn)
    false_positive_rate = fp / (fp + tn)
    true_alerts = volume * prevalence * sensitivity
    false_alerts = volume * (1 - prevalence) * false_positive_rate
    reviews = true_alerts + false_alerts
    return dict(expected_true_alerts=true_alerts, expected_false_alerts=false_alerts,
                expected_missed_signals=volume * prevalence * (1-sensitivity),
                expected_reviews=reviews, review_hours=reviews * minutes_per_review / 60,
                expected_precision=true_alerts/reviews if reviews else None)
