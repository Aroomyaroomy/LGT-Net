import unittest

import numpy as np

from reposition.size_priors import (
    normalize_category,
    prompt_for_room,
    scale_from_prior,
    size_prior,
)


class TestSizePriors(unittest.TestCase):
    def test_headerless_mm_csv(self):
        prior = size_prior("couch")
        self.assertEqual(prior["category"], "couch")
        np.testing.assert_allclose(prior["median"], [1.2573, 0.8372, 0.715], atol=0.02)

    def test_dining_chair_inches(self):
        prior = size_prior("dining chair")
        self.assertEqual(prior["category"], "diningchair")
        self.assertGreater(prior["median"][2], 0.8)
        self.assertLess(prior["median"][2], 0.9)

    def test_aliases_and_room_prompts(self):
        self.assertEqual(normalize_category("TV"), "television")
        self.assertEqual(normalize_category("sofa"), "sofa")
        prompt = prompt_for_room("study_room")
        self.assertIn("office chair", prompt)
        self.assertIn("computer monitor", prompt)
        self.assertIn("file cabinet", prompt)
        self.assertIn("bookcase", prompt)
        self.assertIn("cabinet", prompt_for_room("kitchen"))
        self.assertNotIn(". .", prompt_for_room("bathroom"))
        with self.assertRaises(ValueError):
            prompt_for_room("garage")

    def test_uniform_scale_uses_three_dimensions(self):
        prior = {
            "median": [2.0, 1.0, 0.8],
            "p10": [1.8, 0.9, 0.7],
            "p90": [2.2, 1.1, 0.9],
        }
        factor, dimensions = scale_from_prior([1.0, 0.4, 0.5], prior)
        self.assertAlmostEqual(factor, 2.0)
        np.testing.assert_allclose(dimensions, [2.0, 1.0, 0.8])


if __name__ == "__main__":
    unittest.main()
