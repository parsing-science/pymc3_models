import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
import pymc3 as pm
from pymc3 import summary
from sklearn.linear_model import LogisticRegression as sklearn_LR
from sklearn.model_selection import train_test_split

from pymc3_models.exc import PyMC3ModelsError
from pymc3_models import LogisticRegression


class LogisticRegressionTestCase(unittest.TestCase):
    def setUp(self):
        def numpy_invlogit(x):
            return 1 / (1 + np.exp(-x))

        self.num_pred = 1
        self.num_samples = 10000

        # Set random seed for repeatability
        np.random.seed(27)

        self.alphas = np.random.randn(1)
        self.betas = np.random.randn(1, self.num_pred)
        X = np.random.randn(self.num_samples, self.num_pred)
        Y = np.random.binomial(1, numpy_invlogit(self.alphas[0] + np.sum(self.betas * X, 1)))

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.4)

        self.test_LR = LogisticRegression()
        # Fit the model once
        inference_args = {
            'n': 60000,
            'callbacks': [pm.callbacks.CheckParametersConvergence()]
        }
        # Note: print is here so PyMC3 output won't overwrite the test name
        print('')
        self.test_LR.fit(
            self.X_train,
            self.Y_train,
            num_advi_sample_draws=5000,
            minibatch_size=2000,
            inference_args=inference_args
        )

        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)


class LogisticRegressionFitTestCase(LogisticRegressionTestCase):
    def test_fit_returns_correct_model(self):
        self.assertEqual(self.num_pred, self.test_LR.num_pred)

        np.testing.assert_almost_equal(self.alphas, self.test_LR.trace['alpha'].mean(), decimal=1)
        np.testing.assert_almost_equal(self.betas, self.test_LR.trace['betas'].mean(), decimal=1)


class LogisticRegressionPredictProbaTestCase(LogisticRegressionTestCase):
    def test_predict_proba_returns_probabilities(self):
        probs = self.test_LR.predict_proba(self.X_test)
        self.assertEqual(probs.shape, self.Y_test.shape)

    def test_predict_proba_returns_probabilities_and_std(self):
        probs, stds = self.test_LR.predict_proba(self.X_test, return_std=True)
        self.assertEqual(probs.shape, self.Y_test.shape)
        self.assertEqual(stds.shape, self.Y_test.shape)

    def test_predict_proba_raises_error_if_not_fit(self):
        with self.assertRaises(PyMC3ModelsError) as no_fit_error:
            test_LR = LogisticRegression()
            test_LR.predict_proba(self.X_train)

        expected = 'Run fit on the model before predict.'
        self.assertEqual(str(no_fit_error.exception), expected)


class LogisticRegressionPredictTestCase(LogisticRegressionTestCase):
    def test_predict_returns_predictions(self):
        preds = self.test_LR.predict(self.X_test)
        self.assertEqual(preds.shape, self.Y_test.shape)


class LogisticRegressionScoreTestCase(LogisticRegressionTestCase):
    def test_score_scores(self):
        score = self.test_LR.score(self.X_test, self.Y_test)
        naive_score = np.mean(self.Y_test)
        self.assertGreaterEqual(score, naive_score)

    def test_score_matches_sklearn_performance(self):
        SLR = sklearn_LR()
        SLR.fit(self.X_train, self.Y_train)
        SLR_score = SLR.score(self.X_test, self.Y_test)

        self.test_LR.fit(self.X_train, self.Y_train)
        test_LR_score = self.test_LR.score(self.X_test, self.Y_test)

        self.assertAlmostEqual(SLR_score, test_LR_score, 1)


class LogisticRegressionSaveandLoadTestCase(LogisticRegressionTestCase):
    def test_save_and_load_work_correctly(self):
        probs1 = self.test_LR.predict_proba(self.X_test)
        self.test_LR.save(self.test_dir)

        LR2 = LogisticRegression()

        LR2.load(self.test_dir)

        self.assertEqual(self.test_LR.num_pred, LR2.num_pred)
        self.assertEqual(self.test_LR.num_training_samples, LR2.num_training_samples)
        pd.testing.assert_frame_equal(summary(self.test_LR.trace), summary(LR2.trace))

        probs2 = LR2.predict_proba(self.X_test)

        np.testing.assert_almost_equal(probs2, probs1, decimal=1)
