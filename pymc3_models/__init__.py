import joblib
import matplotlib.pyplot as plt
import pymc3 as pm
import seaborn as sns
from sklearn.base import BaseEstimator


class BayesianModel(BaseEstimator):
    """
    Bayesian model base class
    """
    def __init__(self):
        self.advi_hist = None
        self.advi_trace = None
        self.cached_model = None
        self.num_pred = None
        self.shared_vars = None

    def create_model(self):
        raise NotImplementedError

    def _set_shared_vars(self, shared_vars):
        """
        Sets theano shared variables for the PyMC3 model.
        """
        for key in shared_vars.keys():
            self.shared_vars[key].set_value(shared_vars[key])

    def _inference(self, minibatches, n=200000):
        """
        Runs minibatch variational ADVI and then sample from those results.

        Parameters
        ----------
        minibatches: minibatches for ADVI

        n: number of iterations for ADVI fit, defaults to 200000
        """
        with self.cached_model:
            advi = pm.ADVI()
            approx = pm.fit(
                n=n,
                method=advi,
                more_replacements=minibatches,
                callbacks=[pm.callbacks.CheckParametersConvergence()]
            )

        self.advi_trace = approx.sample(draws=10000)

        self.advi_hist = advi.hist

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def score(self):
        raise NotImplementedError

    def save(self, file_prefix, custom_params=None):
        """
        Saves the advi_trace and custom params to files with the given file_prefix.

        Parameters
        ----------
        file_prefix: str, path and prefix used to identify where to save the trace for this model.
        Ex: given file_prefix = "path/to/file/"
        This will attempt to save to "path/to/file/advi_trace.pickle"

        custom_params: Dictionary of custom parameters to save. Defaults to None
        """
        fileObject = open(file_prefix + 'advi_trace.pickle', 'wb')
        joblib.dump(self.advi_trace, fileObject)
        fileObject.close()

        if custom_params:
            fileObject = open(file_prefix + 'params.pickle', 'wb')
            joblib.dump(custom_params, fileObject)
            fileObject.close()

    def load(self, file_prefix, load_custom_params=False):
        """
        Loads a saved version of the advi_trace, v_params, and custom param files with the given file_prefix.

        Parameters
        ----------
        file_prefix: str, path and prefix used to identify where to load the saved trace for this model.
        Ex: given file_prefix = "path/to/file/"
        This will attempt to load "path/to/file/advi_trace.pickle"

        load_custom_params: Boolean flag to indicate whether custom parameters should be loaded. Defaults to False.

        Returns
        ----------
        custom_params: Dictionary of custom parameters
        """
        self.advi_trace = joblib.load(file_prefix + 'advi_trace.pickle')

        custom_params = None
        if load_custom_params:
            custom_params = joblib.load(file_prefix + 'params.pickle')

        return custom_params

    def plot_elbo(self):
        """
        Plot the ELBO values after running ADVI minibatch.
        """
        sns.set_style("white")
        plt.plot(-self.advi_hist)
        plt.ylabel('ELBO')
        plt.xlabel('iteration')
        sns.despine()
