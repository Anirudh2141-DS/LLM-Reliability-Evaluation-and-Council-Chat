from __future__ import annotations

from collections import Counter

from rlrgf.synthetic import SyntheticDataGenerator


def test_n_ambiguous_is_honored() -> None:
    generator = SyntheticDataGenerator(seed=1)
    test_cases, _ = generator.generate(
        n_normal=1,
        n_adversarial=0,
        n_leakage=0,
        n_jailbreak=0,
        n_unicode=0,
        n_ambiguous=3,
    )

    category_counts = Counter(tc.category for tc in test_cases)
    assert category_counts["ambiguous"] == 3
    assert category_counts["normal"] == 1
