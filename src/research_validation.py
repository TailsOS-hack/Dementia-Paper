"""Shared validation guard for dementia attribution."""
def require(condition, message):
    if not condition:
        raise ValueError(message)
