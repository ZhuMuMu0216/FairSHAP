import numpy as np
import shap
from copy import deepcopy
from src.fairness_metrics.fairness_measures import perturb_numpy_ver
EPSILON = 1e-20

class WeightedExplainer:
    """
    This class provides explanations for model predictions using SHAP values,
    weighted according to a given probability distribution.
    """

    def __init__(self, model, sen_att, priv_val, unpriv_dict):
        """
        Initializes the WeightedExplainer.

        :param model: A machine learning model that supports the predict_proba method.
                      This model is used to predict probabilities which are necessary
                      for SHAP value computation.
        """
        self.model = model
        self.sen_att = sen_att
        self.priv_val = priv_val
        self.unpriv_dict = unpriv_dict

    def explain_instance(
        self, x, y, X_baseline, weights, sample_size=500, shap_sample_size="auto"
    ):
        """
        Generates SHAP values for a single instance using a weighted sample of baseline data.

        :param x: The instance to explain. This should be a single data point.
        :param y: The true label of the instance.
        :param X_baseline: A dataset used as a reference or background distribution.
        :param weights: A numpy array of weights corresponding to the probabilities
                        of choosing each instance in X_baseline.
        :param num_samples: The number of samples to draw from X_baseline to create
                            the background dataset for the SHAP explainer.
        :return: An array of SHAP values for the instance.
        """
        self.y = y
        # Normalize weights to ensure they sum to 1
        weights = weights + EPSILON
        weights = weights / (weights.sum())

        # Generate samples weighted by joint probabilities
        indice = np.random.choice(
            X_baseline.shape[0], p=weights, replace=True, size=sample_size
        )
        indice = np.unique(indice)
        sampled_X_baseline = X_baseline[indice]
        explainer_temp = shap.KernelExplainer(
            self._fairness_value_function, sampled_X_baseline
        )
        shap_values = explainer_temp.shap_values(x, nsamples=shap_sample_size)
        
        return shap_values

    def _fairness_value_function(self, X):
        X_disturbed = perturb_numpy_ver(
            X=X,
            sen_att=self.sen_att,
            priv_val=self.priv_val,
            unpriv_dict=self.unpriv_dict,
            # priv_to_unpriv_map=self.priv_to_unpriv_map,
            ratio=1.0,
        )
        fx = self.model.predict_proba(X)[:, 1]
        fx_q = self.model.predict_proba(X_disturbed)[:, 1]
        # TODO: There might be problems when the classification problem is not binary
        # result = np.abs(fx - fx_q)
        # print(f'result.shape:{result.shape}')
        outputs = np.abs(fx - fx_q)
        return outputs

class FairnessExplainer:
    """
    This class provides SHAP explanations for model predictions across multiple instances,
    using joint probability distributions to weight the baseline data for each instance.
    """

    def __init__(self, model, sen_att, priv_val, unpriv_dict):
        """
        Initializes the FairnessExplainer.

        :param model: A machine learning model that supports the predict_proba method.
                      This model is used to compute SHAP values using weighted baseline data.
        """
        self.model = model
        self.sen_att = sen_att
        self.priv_val = priv_val
        self.unpriv_dict = unpriv_dict
        self.weighted_explainer = WeightedExplainer(
            model, sen_att, priv_val, unpriv_dict
        )

    def _compute_expected_value(self, X_baseline, sample_size):
        """
        Computes the expected value over the baseline dataset.

        :param X_baseline: The baseline dataset.
        :param sample_size: The number of samples to draw from X_baseline.
        :return: The expected value.
        """
        sample_indices = np.random.choice(
            X_baseline.shape[0], size=sample_size, replace=True
        )
        sampled_X_baseline = X_baseline[sample_indices]
        model_output = self.weighted_explainer._fairness_value_function(
            sampled_X_baseline
        )
        return np.mean(model_output)

    def shap_values(
        self,
        X: np.ndarray,
        Y: np.ndarray, # true label of X
        X_baseline: np.ndarray,
        matching,
        sample_size=1000,
        shap_sample_size="auto",
    ):
        """
        Computes SHAP values for multiple instances using a set of joint probability weights.

        :param X: An array of instances to explain. Each instance is a separate data point.
        :param X_baseline: A dataset used as a reference or background distribution.
        :param matching: A matrix of joint probabilities, where each row corresponds to the
                            probabilities for an instance in X, used to weight X_baseline.
        :param num_samples: The number of samples to draw from X_baseline for each instance in X.
        :return: A numpy array of SHAP values for each instance in X.
        """
        self.Y = Y
        # Compute the expected value
        # self.expected_value = self._compute_expected_value(X_baseline, sample_size)

        return np.array(
            [
                self.weighted_explainer.explain_instance(
                    x,
                    y,
                    X_baseline,
                    weights,
                    sample_size=sample_size,
                    shap_sample_size=shap_sample_size,
                )
                for x, y, weights in zip(X, Y, matching)
            ]
        )


