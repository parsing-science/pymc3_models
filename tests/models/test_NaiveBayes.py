import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
from pymc3 import summary
import scipy.stats
from sklearn.model_selection import train_test_split

from pymc3_models import GaussianNaiveBayes
from pymc3_models.exc import PyMC3ModelsError


class GaussianNaiveBayesTestCase(unittest.TestCase):
    def setUp(self):
        """
        Set up a test case with synthetic data.
        """

        self.num_cats = 3
        self.num_pred = 10
        self.num_samples = 100000

        # Set random seed for repeatability
        np.random.seed(27)

        # Generate priors
        self.alpha = np.ones(self.num_cats)
        self.pi = np.random.dirichlet(self.alpha)
        self.mu = np.random.normal(0, 100, size=(self.num_cats, self.num_pred))
        self.sigma = scipy.stats.halfnorm(loc=0, scale=100).rvs(size=(self.num_cats, self.num_pred))
        # Generate data
        Y = np.random.choice(range(self.num_cats), self.num_samples, p=self.pi)
        x_vectors = []
        for i in Y:
            x_vectors.append(np.random.normal(self.mu[i], self.sigma[i]))
        X = np.vstack(x_vectors)

        # Split into train/test sets
        split = train_test_split(X, Y, test_size=0.4)
        self.X_train = split[0]
        self.X_test = split[1]
        self.Y_train = split[2]
        self.Y_test = split[3]

        self.num_training_samples = self.Y_train.shape[0]
        self.test_GNB = GaussianNaiveBayes()
        self.test_dir = tempfile.mkdtemp()

        # Fit the model once and for all
        self.test_GNB.fit(self.X_train, self.Y_train, minibatch_size=2000)

    def tearDown(self):
        """
        Tear down the testing environment.
        """
        shutil.rmtree(self.test_dir)


class GaussianNaiveBayesFitTestCase(GaussianNaiveBayesTestCase):
    def test_fit_returns_correct_model(self):
        """
        Test the model initialization and fit.

        Currently, only the sign of inferred parameters is checked
        against the sign of the parameters used to generate the data.

        TOOO: Find better strategies to test probabilistic code.
        """

        print('')

        # Check that the model correctly infers dimensions
        self.assertEqual(self.num_cats, self.test_GNB.num_cats)
        self.assertEqual(self.num_training_samples, self.test_GNB.num_training_samples)
        self.assertEqual(self.num_pred, self.test_GNB.num_pred)

        # TODO: How do you write tests for a stochastic model?
        # TODO: Diagnose the sampling with a reasonable sampling size?
        np.testing.assert_equal(
            np.sign(self.pi),
            np.sign(self.test_GNB.trace['pi'].mean(axis=0))
        )
        np.testing.assert_equal(
            np.sign(self.sigma),
            np.sign(self.test_GNB.trace['sigma'].mean(axis=0))
        )


class GaussianNaiveBayesPredictProbaTest(GaussianNaiveBayesTestCase):
    def test_predict_proba_returns_probabilities(self):
        print('')
        probs = self.test_GNB.predict_proba(self.X_test)
        self.assertEqual(probs.shape[0], self.Y_test.shape[0])

    def test_predict_proba_raises_error_if_not_fit(self):
        with self.assertRaises(PyMC3ModelsError) as no_fit_error:
            test_GNB = GaussianNaiveBayes()
            test_GNB.predict_proba(self.X_train)
        expected = 'Run fit() on the model before predict()'
        self.assertEqual(str(no_fit_error.exception), expected)


class GaussianNaiveBayesPredictionTestCase(GaussianNaiveBayesTestCase):
    def test_predict_returns_predictions(self):
        """
        Test that the predict() function's  output has the correct shape.
        """
        print('')
        preds = self.test_GNB.predict(self.X_test)
        self.assertEqual(preds.shape, self.Y_test.shape)


@unittest.skip('test not implemented yet')
class GaussianNaiveBayesScoreTestCase(GaussianNaiveBayesTestCase):
    def test_score_scores(self):
        print('')
        # TODO: Figure out how to test the score function
        score = self.test_GNB.score(self.X_test, self.Y_test)


class GaussianNaiveBayesSaveAndLoadTestCase(GaussianNaiveBayesTestCase):
    def test_save_and_load_work_correctly(self):
        print('')
        probs1 = self.test_GNB.predict_proba(self.X_test)
        self.test_GNB.save(self.test_dir)

        GNB2 = GaussianNaiveBayes()
        GNB2.load(self.test_dir)
        self.assertEqual(self.test_GNB.num_cats, GNB2.num_cats)
        self.assertEqual(self.test_GNB.num_pred, GNB2.num_pred)
        self.assertEqual(self.test_GNB.num_training_samples, GNB2.num_training_samples)
        pd.testing.assert_frame_equal(summary(self.test_GNB.trace), summary(GNB2.trace))

        probs2 = GNB2.predict_proba(self.X_test)
        np.testing.assert_almost_equal(probs2, probs1, decimal=1)
