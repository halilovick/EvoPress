import unittest

from scripts.summarize_mistral_fixed_mutation import (
    command_signature,
    jaccard,
)


class MistralFixedMutationSummaryTest(unittest.TestCase):
    def test_jaccard(self) -> None:
        self.assertEqual(jaccard({1, 2}, {1, 2}), 1.0)
        self.assertAlmostEqual(jaccard({1, 2}, {2, 3}), 1 / 3)

    def test_command_signature_ignores_seed_output_and_strength(
        self,
    ) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "command.sh"
            path.write_text(
                "#!/usr/bin/env bash\n"
                "exec python evo_joint_search.py --generations 50 "
                "--seed 2 --max_drop_mutations 1 "
                "--output_dir outputs/example --active_quant_budget\n",
                encoding="utf-8",
            )
            signature, strength = command_signature(path)

        self.assertEqual(strength, 1)
        self.assertEqual(
            signature,
            [
                "python",
                "evo_joint_search.py",
                "--generations",
                "50",
                "--active_quant_budget",
            ],
        )


if __name__ == "__main__":
    unittest.main()
