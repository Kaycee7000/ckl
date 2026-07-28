from typing import Dict, Optional


def abstract_variable(name: str, mapping: Dict[str, str], default: Optional[str] = None) -> str:
    """Map a domain-specific variable name to a domain-agnostic primitive using mapping.

    mapping keys are lowercase substrings to match.
    """
    low = name.lower()
    for key, val in mapping.items():
        if key.lower() in low:
            return val
    return default or name
