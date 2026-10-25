import unittest
import numpy as np
from unittest.mock import patch

from cpaf_ml import (DPLogReg, clip_per_example_gradients, dp_epsilon,
                     dp_verification_fields, rdp_sampled_gaussian)


class DPValidationTests(unittest.TestCase):
    def test_accountant_domain_and_full_sampling(self):
        self.assertAlmostEqual(rdp_sampled_gaussian(1.0, 2.0, 4), 0.5)
        # Independently expanded integer-alpha=2 Poisson Gaussian expression:
        # log(1 + q^2 * (exp(1/sigma^2)-1)).
        q, sigma = .2, 1.7
        expected = np.log1p(q*q * np.expm1(1 / sigma**2))
        self.assertAlmostEqual(rdp_sampled_gaussian(q, sigma, 2), expected, places=12)
        with self.assertRaises(ValueError):
            rdp_sampled_gaussian(0, 1, 2)
        with self.assertRaises(ValueError):
            dp_epsilon(.1, 1, 2, 1)

    def test_dp_model_rejects_zero_noise(self):
        with self.assertRaises(ValueError):
            DPLogReg(0)

    def test_result_verification_status_defaults_to_unverified(self):
        class MissingStatus: pass
        self.assertEqual(dp_verification_fields(MissingStatus()), {
            "verification_status": "EXPERIMENTAL_UNVERIFIED",
            "formal_guarantee_claimed": False})
        model = DPLogReg(sigma=1)
        self.assertEqual(dp_verification_fields(model)["verification_status"],
                         "EXPERIMENTAL_UNVERIFIED")
        self.assertEqual(dp_verification_fields(None), {
            "verification_status": "NOT_APPLICABLE", "formal_guarantee_claimed": False})
        class UnreviewedClaim: verification_status = "VERIFIED_BOUND"
        self.assertEqual(dp_verification_fields(UnreviewedClaim())["verification_status"],
                         "EXPERIMENTAL_UNVERIFIED")

    def test_invalid_training_parameters_and_public_seed_rejection(self):
        for kwargs in [{"sigma": 1, "clip": 0}, {"sigma": 1, "q": 0},
                       {"sigma": 1, "steps": 0}, {"sigma": 1, "lr": -1},
                       {"sigma": 1, "delta": 1},
                       {"sigma": 1, "seed": 42}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DPLogReg(**kwargs)

    def test_accountant_composition_and_noise_direction(self):
        one = dp_epsilon(1.0, 2.0, 1, .01)
        two = dp_epsilon(1.0, 2.0, 2, .01)
        repeated = dp_epsilon(1.0, 2.0, 1, .01, num_runs=2)
        more_noise = dp_epsilon(.1, 2.0, 10, .01)
        less_noise = dp_epsilon(.1, 1.0, 10, .01)
        self.assertGreater(two, one)
        self.assertAlmostEqual(two, repeated)
        self.assertGreater(less_noise, more_noise)
        with self.assertRaises(ValueError):
            dp_epsilon(.1, 1, 1, .01, num_runs=0)

    def test_dp_fit_validates_shapes_and_finiteness(self):
        model = DPLogReg(sigma=1, q=1, steps=2, lr=.1)
        X = np.array([[0., 1.], [1., 0.], [.5, .5], [1., 1.]])
        y = np.array([0, 1, 0, 1])
        model.fit(X, y)
        self.assertTrue(np.isfinite(model.epsilon_unverified))
        self.assertGreaterEqual(model.epsilon_unverified, 0.0)
        self.assertFalse(model.accounting_status.startswith("validated"))
        self.assertEqual(model.randomness_source,
                         "secrets.SystemRandom (OS-backed; not seeded from experiment seed)")
        self.assertEqual(model.delta, 1e-5)
        with self.assertRaises(ValueError):
            DPLogReg(sigma=1, q=1, steps=1).fit(X, [0])
        with self.assertRaises(ValueError):
            DPLogReg(sigma=1, q=1, steps=1).fit(np.full_like(X, np.nan), y)
        with self.assertRaises(ValueError):
            DPLogReg(sigma=1, q=1, steps=1).fit(np.empty((0, 2)), np.empty(0))

    def test_per_example_l2_clipping(self):
        clipped = clip_per_example_gradients([[3., 4.], [0.1, 0.]], 1.0)
        np.testing.assert_allclose(np.linalg.norm(clipped, axis=1), [1.0, 0.1])
        with self.assertRaises(ValueError):
            clip_per_example_gradients([[1., np.nan]], 1.0)

    def test_dp_training_uses_system_random_not_numpy_seed(self):
        class DeterministicSystemRandom:
            def random(self): return .25
            def gauss(self, mu, sigma): return mu + sigma
        X = np.array([[0.0], [1.0], [0.5]])
        y = np.array([0, 1, 0])
        with patch("cpaf_ml.secrets.SystemRandom", DeterministicSystemRandom):
            model = DPLogReg(sigma=1, q=1, steps=2, lr=.1).fit(X, y)
        self.assertTrue(np.isfinite(model.w).all())
        self.assertEqual(model.randomness_source.split(" ")[0], "secrets.SystemRandom")
        self.assertEqual(model.verification_status, "EXPERIMENTAL_UNVERIFIED")

    def test_noise_changes_parameters_and_test_mock_is_reproducible(self):
        class FixedNoise:
            def __init__(self, value): self.value = value
            def random(self): return .25
            def gauss(self, mu, sigma): return self.value
        X = np.array([[0.0], [1.0], [.5]])
        y = np.array([0, 1, 0])
        def fit_with(value):
            with patch("cpaf_ml.secrets.SystemRandom", lambda: FixedNoise(value)):
                return DPLogReg(sigma=1, q=1, steps=2, lr=.1).fit(X, y)
        a, b = fit_with(.25), fit_with(.25)
        np.testing.assert_array_equal(a.w, b.w)
        c = fit_with(0.0)
        self.assertFalse(np.allclose(a.w, c.w))


if __name__ == "__main__":
    unittest.main()
