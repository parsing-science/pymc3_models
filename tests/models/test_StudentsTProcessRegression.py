import unittest
import shutil
import tempfile

import numpy as np
import pandas as pd
import pymc3 as pm
from pymc3 import summary
from sklearn.gaussian_process import GaussianProcessRegressor as skGaussianProcessRegressor
from sklearn.model_selection import train_test_split


from pymc3_models.exc import PyMC3ModelsError
from pymc3_models.models.StudentsTProcessRegression import StudentsTProcessRegression


class StudentsTProcessRegressionTestCase(unittest.TestCase):

    def setUp(self):
        self.num_training_samples = 150
        self.num_pred = 1

        self.length_scale = 2.0
        self.noise_variance = 1.0
        self.signal_variance = 1.0
        self.degrees_of_freedom = 3.0

        X = np.linspace(start=0, stop=10, num=self.num_training_samples)[:, None]
        cov_func = self.signal_variance**2 * pm.gp.cov.ExpQuad(self.num_pred,
                                                               self.length_scale)

        mean_func = pm.gp.mean.Zero()
        f_ = np.random.multivariate_normal(mean_func(X).eval(),
                                           cov_func(X).eval() + 1e-8 * np.eye(self.num_training_samples),
                                           self.num_pred
                                           ).flatten()

        y = f_ + self.noise_variance * np.random.standard_t(self.degrees_of_freedom,
                                                            size=self.num_training_samples)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3
        )

        self.test_STPR = StudentsTProcessRegression()
        # self.test_nuts_STPR = StudentsTProcessRegression()
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)


class StudentsTProcessRegressionFitTestCase(StudentsTProcessRegressionTestCase):
    def test_advi_fit_returns_correct_model(self):
        # This print statement ensures PyMC3 output won't overwrite the test name
        print('')
        self.test_STPR.fit(self.X_train, self.y_train)

        self.assertEqual(self.num_pred, self.test_STPR.num_pred)
        self.assertAlmostEqual(self.signal_variance,
                               int(self.test_STPR.summary['mean']['signal_variance__0']),
                               0)
        self.assertAlmostEqual(self.length_scale,
                               int(self.test_STPR.summary['mean']['length_scale__0_0']),
                               0)
        self.assertAlmostEqual(self.noise_variance,
                               int(self.test_STPR.summary['mean']['noise_variance__0']),
                               0)

    # def test_nuts_fit_returns_correct_model(self):
    #     # This print statement ensures PyMC3 output won't overwrite the test name
    #     print('')
    #     self.test_nuts_STPR.fit(self.X_train, self.y_train, inference_type='nuts')
    #
    #     self.assertEqual(self.num_pred, self.test_nuts_STPR.num_pred)
    #     self.assertAlmostEqual(self.signal_variance,
    #                            int(self.test_nuts_STPR.summary['mean']['signal_variance__0']),
    #                            0)
    #     self.assertAlmostEqual(self.length_scale,
    #                            int(self.test_nuts_STPR.summary['mean']['length_scale__0_0']),
    #                            0)
    #     self.assertAlmostEqual(self.noise_variance,
    #                            int(self.test_nuts_STPR.summary['mean']['noise_variance__0']),
    #                            0)


class StudentsTProcessRegressionPredictTestCase(StudentsTProcessRegressionTestCase):
    def test_predict_returns_predictions(self):
        print('')
        self.test_STPR.fit(self.X_train, self.y_train)
        preds = self.test_STPR.predict(self.X_test)
        self.assertEqual(self.y_test.shape, preds.shape)

    def test_predict_returns_mean_predictions_and_std(self):
        print('')
        self.test_STPR.fit(self.X_train, self.y_train)
        preds, stds = self.test_STPR.predict(self.X_test, return_std=True)
        self.assertEqual(self.y_test.shape, preds.shape)
        self.assertEqual(self.y_test.shape, stds.shape)

    def test_predict_raises_error_if_not_fit(self):
        print('')
        with self.assertRaises(PyMC3ModelsError) as no_fit_error:
            test_STPR = StudentsTProcessRegression()
            test_STPR.predict(self.X_train)

        expected = 'Run fit on the model before predict.'
        self.assertEqual(str(no_fit_error.exception), expected)


class StudentsTProcessRegressionScoreTestCase(StudentsTProcessRegressionTestCase):
    def test_score_matches_sklearn_performance(self):
        print('')
        skGPR = skGaussianProcessRegressor()
        skGPR.fit(self.X_train, self.y_train)
        skGPR_score = skGPR.score(self.X_test, self.y_test)

        self.test_STPR.fit(self.X_train, self.y_train)
        test_STPR_score = self.test_STPR.score(self.X_test, self.y_test)

        self.assertAlmostEqual(skGPR_score, test_STPR_score, 1)


class StudentsTProcessRegressionSaveAndLoadTestCase(StudentsTProcessRegressionTestCase):
    def test_save_and_load_work_correctly(self):
        print('')
        self.test_STPR.fit(self.X_train, self.y_train)
        score1 = self.test_STPR.score(self.X_test, self.y_test)
        self.test_STPR.save(self.test_dir)

        STPR2 = StudentsTProcessRegression()
        STPR2.load(self.test_dir)

        self.assertEqual(self.test_STPR.inference_type, STPR2.inference_type)
        self.assertEqual(self.test_STPR.num_pred, STPR2.num_pred)
        self.assertEqual(self.test_STPR.num_training_samples, STPR2.num_training_samples)
        pd.testing.assert_frame_equal(summary(self.test_STPR.trace),
                                      summary(STPR2.trace))

        score2 = STPR2.score(self.X_test, self.y_test)
        self.assertAlmostEqual(score1, score2, 1)
