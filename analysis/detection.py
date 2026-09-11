"""Shared keyword baseline. Matches proxy vocabulary, not a clinical diagnosis."""

def rule_classify(text, drug_name, keyword_dict):
    if not text:
        return 0, []
    combined = (str(text) + ' ' + str(drug_name or '')).lower()
    fired = sorted(category for category, keywords in keyword_dict.items()
                   if any(kw in combined for kw in keywords))
    return int(bool(fired)), fired
