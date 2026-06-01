import random
import unittest

from scripts.build_depth_baseline_config import build_baseline_config, count_removed_modules


REFERENCE_CONFIG = ["none", "mlp", "attn", "attn+mlp"]


class BuildDepthBaselineConfigTest(unittest.TestCase):
    def test_late_layer_matches_counts(self) -> None:
        config = build_baseline_config(REFERENCE_CONFIG, method="late_layer", seed=1)
        self.assertEqual(config, ["none", "none", "attn+mlp", "attn+mlp"])
        self.assertEqual(count_removed_modules(config), (2, 2))

    def test_early_layer_can_protect_layer_zero(self) -> None:
        config = build_baseline_config(
            REFERENCE_CONFIG,
            method="early_layer",
            seed=1,
            protect_layer_zero=True,
        )
        self.assertEqual(config, ["none", "attn+mlp", "attn+mlp", "none"])
        self.assertEqual(count_removed_modules(config), (2, 2))

    def test_random_is_deterministic_and_matches_counts(self) -> None:
        first = build_baseline_config(REFERENCE_CONFIG, method="random", seed=3)
        second = build_baseline_config(REFERENCE_CONFIG, method="random", seed=3)
        self.assertEqual(first, second)
        self.assertEqual(count_removed_modules(first), (2, 2))

        rng = random.Random(3)
        expected_attn = sorted(rng.sample(range(4), 2))
        expected_mlp = sorted(rng.sample(range(4), 2))
        actual_attn = [index for index, value in enumerate(first) if value in {"attn", "attn+mlp"}]
        actual_mlp = [index for index, value in enumerate(first) if value in {"mlp", "attn+mlp"}]
        self.assertEqual(actual_attn, expected_attn)
        self.assertEqual(actual_mlp, expected_mlp)


if __name__ == "__main__":
    unittest.main()
