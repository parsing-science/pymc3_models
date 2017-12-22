import numpy as np
import pymc3 as pm
from sklearn.metrics import accuracy_score
import theano
import theano.tensor as T

from pymc3_models.exc import PyMC3ModelsError
from pymc3_models import BayesianModel


class HLR(BayesianModel):
    """
    Custom Hierachical Logistic Regression built using PyMC3.
    """

    def __init__(self):
        super(HLR, self).__init__()
        self.num_cats = None

    def create_model(self):
        """
        Creates and returns the PyMC3 model.

        Returns the model.
        """
        model_input = theano.shared(np.zeros([1, self.num_pred]))

        model_output = theano.shared(np.zeros(1))

        model_cats = theano.shared(np.zeros(1, dtype='int'))

        self.shared_vars = {
            'model_input': model_input,
            'model_output': model_output,
            'model_cats': model_cats
        }

        model = pm.Model()

        with model:
            mu_alpha = pm.Normal('mu_alpha', mu=0, sd=100)
            sigma_alpha = pm.HalfNormal('sigma_alpha', sd=100)

            mu_beta = pm.Normal('mu_beta', mu=0, sd=100)
            sigma_beta = pm.HalfNormal('sigma_beta', sd=100)

            alpha = pm.Normal('alpha', mu=mu_alpha, sd=sigma_alpha, shape=(self.num_cats,))
            beta = pm.Normal('beta', mu=mu_beta, sd=sigma_beta, shape=(self.num_cats, self.num_pred))

            c = model_cats

            temp = alpha[c] + T.sum(beta[c] * model_input, 1)

            p = pm.invlogit(temp)

            o = pm.Bernoulli('o', p, observed=model_output)

        return model

    def fit(self, X, y, cats, n=200000, batch_size=100):
        """
        Train the HLR model

        Parameters
        ----------
        X : numpy array, shape [n_samples, n_features]

        y : numpy array, shape [n_samples, ]

        cats: numpy array, shape [n_samples, ]

        n: number of iterations for ADVI fit, defaults to 200000

        batch_size: number of samples to include in each minibatch for ADVI, defaults to 100
        """
        self.num_cats = len(np.unique(cats))
        num_samples, self.num_pred = X.shape

        if self.cached_model is None:
            self.cached_model = self.create_model()

        with self.cached_model:

            minibatches = {
                self.shared_vars['model_input']: pm.Minibatch(X, batch_size=batch_size),
                self.shared_vars['model_output']: pm.Minibatch(y, batch_size=batch_size),
                self.shared_vars['model_cats']: pm.Minibatch(cats, batch_size=batch_size)
            }

            self._inference(minibatches, n)

        return self

    def predict_proba(self, X, cats, return_std=False):
        """
        Predicts probabilities of new data with a trained HLR

        Parameters
        ----------
        X : numpy array, shape [n_samples, n_features]

        cats: numpy array, shape [n_samples, ]

        return_std: Boolean flag of whether to return standard deviations with mean probabilities. Defaults to False.
        """

        if self.advi_trace is None:
            raise PyMC3ModelsError('Run fit on the model before predict.')

        num_samples = X.shape[0]

        if self.cached_model is None:
            self.cached_model = self.create_model()

        self._set_shared_vars({'model_input': X, 'model_output': np.zeros(num_samples), 'model_cats': cats})

        ppc = pm.sample_ppc(self.advi_trace, model=self.cached_model, samples=2000)

        if return_std:
            return ppc['o'].mean(axis=0), ppc['o'].std(axis=0)
        else:
            return ppc['o'].mean(axis=0)

    def predict(self, X, cats):
        """
        Predicts labels of new data with a trained model

        Parameters
        ----------
        X : numpy array, shape [n_samples, n_features]

        cats: numpy array, shape [n_samples, ]
        """
        ppc_mean = self.predict_proba(X, cats)

        pred = ppc_mean > 0.5

        return pred

    def score(self, X, y, cats):
        """
        Scores new data with a trained model.

        Parameters
        ----------
        X : numpy array, shape [n_samples, n_features]

        y : numpy array, shape [n_samples, ]

        cats: numpy array, shape [n_samples, ]
        """

        return accuracy_score(y, self.predict(X, cats))

    def save(self, file_prefix):
        params = {'num_cats': self.num_cats, 'num_pred': self.num_pred}

        super(HLR, self).save(file_prefix, params)

    def load(self, file_prefix):
        params = super(HLR, self).load(file_prefix, load_custom_params=True)

        self.num_cats = params['num_cats']
        self.num_pred = params['num_pred']
