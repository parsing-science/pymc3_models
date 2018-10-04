import numpy as np
import pymc3 as pm
from sklearn.metrics import r2_score
import theano
import theano.tensor as T

from pymc3_models.exc import PyMC3ModelsError
from pymc3_models.models.GaussianProcessRegression import GaussianProcessRegression


class StudentsTProcessRegression(GaussianProcessRegression):
    """
    StudentsT Process Regression built using PyMC3.
    """

    def __init__(self, prior_mean=0.0):
        super(StudentsTProcessRegression, self).__init__(prior_mean=prior_mean)

    def create_model(self):
        """
        Creates and returns the PyMC3 model.

        Note: The size of the shared variables must match the size of the
        training data. Otherwise, setting the shared variables later will raise
        an error. See http://docs.pymc.io/advanced_theano.html

        Returns
        ----------
        the PyMC3 model
        """
        model_input = theano.shared(np.zeros([self.num_training_samples, self.num_pred]))

        model_output = theano.shared(np.zeros(self.num_training_samples))

        self.shared_vars = {
            'model_input': model_input,
            'model_output': model_output,
        }

        self.gp = None
        model = pm.Model()

        with model:
            length_scale = pm.Gamma('length_scale', alpha=2, beta=0.5, shape=(1, self.num_pred))
            signal_variance = pm.HalfCauchy('signal_variance', beta=2, shape=(1))
            noise_variance = pm.HalfCauchy('noise_variance', beta=2, shape=(1))
            degrees_of_freedom = pm.Gamma('degrees_of_freedom', alpha=2, beta=0.1, shape=(1))

            # cov_function = signal_variance**2 * pm.gp.cov.ExpQuad(1, length_scale)
            cov_function = signal_variance ** 2 * pm.gp.cov.Matern52(1, length_scale)

            # mean_function = pm.gp.mean.Zero()
            mean_function = pm.gp.mean.Constant(self.prior_mean)

            self.gp = pm.gp.Latent(mean_func=mean_function, cov_func=cov_function)

            f = self.gp.prior('f', X=model_input.get_value())

            y = pm.StudentT('y', mu=f, lam=1 / signal_variance,
                            nu=degrees_of_freedom, observed=model_output)

        return model

    def save(self, file_prefix):
        params = {
            'inference_type': self.inference_type,
            'num_pred': self.num_pred,
            'num_training_samples': self.num_training_samples
        }

        super(StudentsTProcessRegression, self).save(file_prefix, params)

    def load(self, file_prefix):
        params = super(StudentsTProcessRegression, self).load(file_prefix, load_custom_params=True)

        self.inference_type = params['inference_type']
        self.num_pred = params['num_pred']
        self.num_training_samples = params['num_training_samples']
