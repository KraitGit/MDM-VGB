import unittest

from algorithms.algorithm_utils import length_max_steps
from algorithms.config import apply_budget_overrides, value_guidance_config
from algorithms.vgb_terminal import budgeted_max_steps


class BudgetOverrideTest(unittest.TestCase):
    def test_bon_n_sets_candidate_count(self):
        config = {"algorithms": {"BoN": {"N": 8}, "value_guidance": {"max_steps_multiplier": 8}}}

        apply_budget_overrides(config, "BoN", n=4)

        self.assertEqual(value_guidance_config(config, "BoN")["N"], 4)

    def test_vgb_n_sets_step_multiplier(self):
        config = {"algorithms": {"value_guidance": {"max_steps_multiplier": 8}}}

        apply_budget_overrides(config, "VGB", n=16)

        resolved = value_guidance_config(config, "VGB")
        self.assertEqual(resolved["max_steps_multiplier"], 16)
        self.assertNotIn("N", resolved)
        self.assertEqual(length_max_steps(resolved, [[0] * 89]), 1424)

    def test_vgb_momentum_n_sets_step_multiplier(self):
        config = {"algorithms": {"value_guidance": {"max_steps_multiplier": 8}}}

        apply_budget_overrides(config, "VGB-Momentum", n=2)

        resolved = value_guidance_config(config, "VGB-Momentum")
        self.assertEqual(resolved["max_steps_multiplier"], 2)
        self.assertEqual(budgeted_max_steps(resolved, None, [None], [[0] * 89], None), [178])

    def test_matching_aliases_are_allowed(self):
        config = {"algorithms": {"value_guidance": {"max_steps_multiplier": 8}}}

        apply_budget_overrides(config, "VGB", n=16, max_steps_multiplier=16)

        self.assertEqual(value_guidance_config(config, "VGB")["max_steps_multiplier"], 16)

    def test_conflicting_aliases_are_rejected(self):
        config = {"algorithms": {"value_guidance": {"max_steps_multiplier": 8}}}

        with self.assertRaisesRegex(ValueError, "must match"):
            apply_budget_overrides(config, "VGB", n=8, max_steps_multiplier=16)

    def test_n_is_rejected_for_vgr(self):
        config = {"algorithms": {"value_guidance": {"max_steps_multiplier": 8}}}

        with self.assertRaisesRegex(ValueError, "only supported"):
            apply_budget_overrides(config, "VGR", n=8)

    def test_budget_values_must_be_positive(self):
        config = {"algorithms": {"value_guidance": {"max_steps_multiplier": 8}}}

        with self.assertRaisesRegex(ValueError, "positive"):
            apply_budget_overrides(config, "VGB", n=0)


if __name__ == "__main__":
    unittest.main()
