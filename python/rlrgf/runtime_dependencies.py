"""
Runtime dependency checks for local inference backends.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util


REQUIRED_LOCAL_DEPENDENCIES = {
    "torch": "pip install torch",
    "transformers": "pip install transformers",
    "accelerate": "pip install accelerate",
}

OPTIONAL_LOCAL_DEPENDENCIES = {
    "bitsandbytes": "pip install bitsandbytes",
    "sentencepiece": "pip install sentencepiece",
    "safetensors": "pip install safetensors",
}


@dataclass(frozen=True)
class DependencyReport:
    missing_required: dict[str, str]
    missing_optional: dict[str, str]

    @property
    def ok(self) -> bool:
        return not self.missing_required

    def format_error(self) -> str:
        if self.ok:
            return ""
        parts = [
            f"Missing dependency: {name}. Install with `{hint}`."
            for name, hint in self.missing_required.items()
        ]
        return " ".join(parts)


def _find_missing(requirements: dict[str, str]) -> dict[str, str]:
    missing: dict[str, str] = {}
    for module_name, install_hint in requirements.items():
        if importlib.util.find_spec(module_name) is None:
            missing[module_name] = install_hint
    return missing


def validate_local_backend_dependencies(strict: bool = False) -> DependencyReport:
    report = DependencyReport(
        missing_required=_find_missing(REQUIRED_LOCAL_DEPENDENCIES),
        missing_optional=_find_missing(OPTIONAL_LOCAL_DEPENDENCIES),
    )
    if strict and not report.ok:
        raise RuntimeError(report.format_error())
    return report
