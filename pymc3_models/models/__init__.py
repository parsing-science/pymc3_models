import joblib
import matplotlib.pyplot as plt
import pymc3 as pm
import seaborn as sns
from sklearn.base import BaseEstimator

from pymc3_models.exc import PyMC3ModelsError


class BayesianModel(BaseEstimator):
    """
    Bayesian model base class
    """
    def __init__(self):
        self.cached_model = None
        self.default_advi_sample_draws = 10000
        self.inference_type = None
        self.num_pred = None
        self.shared_vars = None
        self.summary = None
        self.trace = None

    def create_model(self):
        raise NotImplementedError

    def _set_shared_vars(self, shared_vars):
        """
        Sets theano shared variables for the PyMC3 model.
        """
        for key in shared_vars.keys():
            self.shared_vars[key].set_value(shared_vars[key])

    def _inference(self, inference_type='advi', inference_args=None):
        """
        Calls internal methods for two types of inferences. Raises an error if the inference_type is not supported.

        Parameters
        ==========
        inference_type : string, specifies which inference method to call. Defaults to 'advi'. Currently, only 'advi' and 'nuts' are supported

        inference_args : dict, arguments to be passed to the inference methods. Check the PyMC3 docs to see what is permitted. Defaults to None.
        """
        if inference_type == 'advi':
            self._advi_inference(inference_args)
        elif inference_type == 'nuts':
            self._nuts_inference(inference_args)
        else:
            raise PyMC3ModelsError('{} is not a supported type of inference'.format(inference_type))

    def _advi_inference(self, inference_args):
        """
        Runs variational ADVI and then samples from those results.

        Parameters
        ----------
        inference_args : dict, arguments to be passed to the PyMC3 fit method. See PyMC3 doc for permissible values.
        """
        with self.cached_model:
            inference = pm.ADVI()
            approx = pm.fit(method=inference, **inference_args)

        self.approx = approx
        self.trace = approx.sample(draws=self.default_advi_sample_draws)
        self.summary = pm.df_summary(self.trace)
        self.advi_hist = inference.hist

    def _nuts_inference(self, inference_args):
        """
        Runs NUTS inference.

        Parameters
        ----------
        inference_args : dict, arguments to be passed to the PyMC3 sample method. See PyMC3 doc for permissible values.
        """
        with self.cached_model:
            step = pm.NUTS()
            nuts_trace = pm.sample(step=step, **inference_args)

        self.trace = nuts_trace
        self.summary = pm.df_summary(self.trace)

    def _set_default_inference_args(self):
        """
        Set default values for inference arguments if none are provided, dependent on inference type.

        ADVI
        -----
        callbacks : list containing a parameter stopping check.

        n : number of iterations for ADVI fit, defaults to 200000

        NUTS
        -----
        draws : the number of samples to draw, defaults to 2000
        """
        if self.inference_type == 'advi':
            inference_args = {
                'n': 200000,
                'callbacks': [pm.callbacks.CheckParametersConvergence()]
            }
        elif self.inference_type == 'nuts':
            inference_args = {
                'draws': 2000
            }
        else:
            inference_args = None

        return inference_args

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def score(self):
        raise NotImplementedError

    def save(self, file_prefix, custom_params=None):
        """
        Saves the trace and custom params to files with the given file_prefix.

        Parameters
        ----------
        file_prefix : str, path and prefix used to identify where to save the trace for this model.
            Ex: given file_prefix = "path/to/file/"
            This will attempt to save to "path/to/file/trace.pickle"

        custom_params : Dictionary of custom parameters to save. Defaults to None
        """
        fileObject = open(file_prefix + 'trace.pickle', 'wb')
        joblib.dump(self.trace, fileObject)
        fileObject.close()

        if custom_params:
            fileObject = open(file_prefix + 'params.pickle', 'wb')
            joblib.dump(custom_params, fileObject)
            fileObject.close()

    def load(self, file_prefix, load_custom_params=False):
        """
        Loads a saved version of the trace, and custom param files with the given file_prefix.

        Parameters
        ----------
        file_prefix : str, path and prefix used to identify where to load the saved trace for this model.
            Ex: given file_prefix = "path/to/file/"
            This will attempt to load "path/to/file/trace.pickle"

        load_custom_params : Boolean flag to indicate whether custom parameters should be loaded. Defaults to False.

        Returns
        ----------
        custom_params : Dictionary of custom parameters
        """
        self.trace = joblib.load(file_prefix + 'trace.pickle')

        custom_params = None
        if load_custom_params:
            custom_params = joblib.load(file_prefix + 'params.pickle')

        return custom_params

    def plot_elbo(self):
        """
        Plot the ELBO values after running ADVI minibatch.
        """
        if self.inference_type != 'advi':
            raise PyMC3ModelsError(
                'This method should only be called after calling fit with ADVI minibatch.'
            )

        sns.set_style("white")
        plt.plot(-self.advi_hist)
        plt.ylabel('ELBO')
        plt.xlabel('iteration')
        sns.despine()
