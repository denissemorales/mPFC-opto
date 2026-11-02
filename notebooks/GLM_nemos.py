import copy
import gc
import time
import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

try:
    import nemos as nmo
except ImportError:
    nmo = None
    warnings.warn(
        "nemos does not exist. Skipping nemos import. Fitting is disabled."
    )
import numpy as np

# from jaxlib.xla_extension import Array
from sklearn.model_selection import GroupKFold, KFold


class PopulationGLM_CV:
    def __init__(
        self,
        activation="exp",
        loss_type="poisson",
        regularization="ridge",
        lambda_series=10.0 ** np.linspace(-1, -8, 30),
        n_folds=5,
        auto_split=True,
        split_by_group=True,
        split_random_state=None,
        solver_kwargs=None,
    ):
        """
        PopulationGLM_CV class
        It fits GLM with n fold cross validation using PopulationGLM function from nemos,
        selects proper regularization values for each response based on deviance of CV held-out data,
        and returns weights and intercepts from models re-fitted with all datapoints in training data with selected regularization values.
        Fit Y with multiple responses simultaneously.

        Model: Y = activation(X * w + w0) + noise
        X: n_samples x n_features
        Y: n_samples x n_responses
        w: n_features x n_responses
        w0: n_responses

        Currently only Poisson GLM and Gamma GLM are supported with the following arguments:
        Poisson GLM: activation = 'exp' or 'softplus', loss_type = 'poisson'
        Gamma GLM: activation = 'softplus', loss_type = 'gamma'
        ## NOTE that although Gamma GLM is implemented, it hasn't been tested. Use it with caution!! ##

        Input parameters::
        activation: {'exp', 'softplus'}, default = 'exp'
        loss_type: {'poisson', 'gamma'}, default = 'poisson'
        regularization: {'rigde', 'lasso', 'group_lasso'}, default = 'ridge'
        lambda_series: list or ndarray of a series of regularization strength (lambda), in descending order,
                       default = 10.0 ** np.linspace(-1, -8, 30)
        n_folds: number of CV folds, default = 5
        auto_split: perform CV split automatically or not, bool, default = True
        split_by_group: perform CV split according to a third-party provided group when auto_split = True, default = True
        split_random_state: optional, numpy random state for CV splitting
        solver_kwargs: solver_kwargs for jaxopt optimizer

        Attributes::
        Key attributes you might look up after fitting and model selection:
          selected_w0, selected_w, selected_lambda, selected_lambda_ind, selected_frac_dev_expl_cv

        List of attributes:
        act_func: activation function based on input activation type
        observation_model: nemos observation model based on the act_func and loss_type
        fitted: if the model has been fitted, bool
        selected: if model selection has been performed, bool
        n_features: number of features seen during fit (X.shape[1])
        n_responses: number of responses seen during fit (Y.shape[1])
        n_lambdas: number of regularization strengths (lambda_series.shape[0])
        selected_w0: intercepts for selected models (from the re-fitted models using all datapoints), ndarray of shape (n_responses,)
        selected_w: weights for selected models (from the re-fitted models using all datapoints),
                    ndarray of shape (n_features, n_responses)
        selected_lambda: lambda values for each response for selected models, ndarray of shape (n_responses,)
        selected_lambda_ind: indices of lambdas for each response for selected models, ndarray of shape (n_responses,)
        w_series_dict: all fitted intercepts and weights for all lambdas across all folds,
                       dictionary arranged as {n_fold: [[w0, w] for lambda 1, [w0, w] for lambda 2,...]} for fold 0 to n_folds-1,
                       additionally, w_series_dict[n_folds] is the w_series when fitting with all datapoints
        selected_frac_dev_expl_cv: fraction deviance explained evaluated on cv held-out data for selected models
        split_random_state: numpy random state for CV splitting, either given by user or generated automatically
        train_idx: indices for training data for each fold, either given by user or generated through auto_split,
                   dictionary arranges as {n_fold: ndarray containing training indices of that fold for n_fold in range(n_folds)},
                   additionally, train_idx[n_folds] returns the indices for all datapoints which is used for a final fit
        val_idx: indices for CV held-out data for each fold, either given by user or generated through auto_split,
                 dictionary arranges as {n_fold: ndarray containing validation indices of that fold for n_fold in range(n_folds)}
        Y_fit: values of response matrix used in fitting; used in model selection
        all_prediction: prediction made on CV held-out data for each fold during fitting; used in model selection
        all_deviance: model deviance computed on CV held-out data for each fold during fitting; used in model selection

        Methods::
        fit(X, Y, [initial_w0, initial_w, feature_group_size, verbose]):
            fit GLM to training data with cross validation
        select_model([se_fraction, min_lambda, make_fig]): select model after fit is called
        predict(X): returns prediction on input data X using selected models after select_model is called; inherited from GLM class
        evaluate(X, Y, [make_fig]): compute fraction deviance explained on input data X, Y using selected models
                                    after select_model is called; inherited from GLM class
        """

        self.activation = activation
        self.loss_type = loss_type
        self.regularization = regularization
        self.lambda_series = np.sort(np.array(lambda_series))[
            ::-1
        ]  # make sure to fit starts with the largest regularization
        self.n_lambdas = self.lambda_series.shape[0]
        self.n_folds = n_folds
        self.auto_split = auto_split
        self.split_by_group = split_by_group

        # set up activation function
        if self.activation == "exp":
            self.act_func = jnp.exp
        elif self.activation == "softplus":
            self.act_func = jax.nn.softplus

        # set up observation model
        # NOTE: newer nemos (>=0.2.x) moved `inverse_link_function` off the
        # observation-model constructor and onto `nemos.glm.PopulationGLM` itself
        # (see `_fit` below, where it's passed at model-construction time instead).
        if self.loss_type == "poisson":
            self.observation_model = nmo.observation_models.PoissonObservations()
        elif self.loss_type == "gamma":
            self.observation_model = nmo.observation_models.GammaObservations()

        # if no auto-split, change split_by_group to False
        if self.auto_split == False:
            self.split_by_group = False

        if split_random_state is not None:
            self.split_random_state = split_random_state
        else:
            self.split_random_state = np.random.randint(0, high=2**31)

        # set solver_kwargs
        if solver_kwargs is not None:
            self.solver_kwargs = solver_kwargs
        else:
            self.solver_kwargs = {"tol": 1e-3, "maxiter": 1000}

    def _cv_split(self, X, Y, group_idx=None):
        """
        <Function> Split data into train-validation set for different CV folds, used by <Method> fit
        Input parameters::
        X: design matrix, ndarray of shape (n_samples, n_features)
        Y: response matrix, ndarray of shape (n_samples, n_responses)
        group_idx: third-party provided group for each sample for split_by_group = True, ndarray of shape (n_samples, )

        Returns::
        train_idx: indices for training data for each fold,
                   dictionary arranges as {n_fold: ndarray containing training indices for that fold} for fold 0 to n_folds-1,
                   additionally, train_idx[n_folds] returns the indices for all datapoints which is used for a final fit
        val_idx: indices for CV held-out data for each fold,
                 dictionary arranges as {n_fold: ndarray containing validation indices for that fold} for fold 0 to n_folds-1
        """

        # choose splitter
        if self.split_by_group:
            assert (
                group_idx is not None
            ), "Error: You must supply group_idx if split_by_group = True"
            assert (
                group_idx.shape[0] == Y.shape[0]
            ), "Error: Number of timepoints (axis 0) of group_idx and data not matching!"
            np.random.seed(self.split_random_state)
            splitter = GroupKFold(n_splits=self.n_folds).split(X, Y, group_idx)
        else:
            splitter = KFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.split_random_state,
            ).split(X, Y)

        # create dictionaries to hold train and validation idx for each fold
        train_idx = {}
        val_idx = {}

        # split data and save the train and validation index
        for n_fold, (train_index, val_index) in enumerate(splitter):
            train_idx[n_fold] = train_index
            val_idx[n_fold] = val_index

        # we fit the model to all datapoints at the end and report weights of this fit after model selection
        train_idx[self.n_folds] = np.arange(Y.shape[0])

        return train_idx, val_idx

    def fit(
        self,
        X,
        Y,
        train_idx=None,
        val_idx=None,
        group_idx=None,
        initial_w=None,
        initial_w0=None,
        feature_group_mask=None,
        verbose=True,
    ):
        """
        <Method> Fit GLM_CV. This method overwrites the fit method in GLM class.
                 This method loops over each CV fold to perform fitting + one final round of fitting using all datapoints;
                 it also saves the prediction and model deviance on CV held-out data that are used in model selection process.
        Input parameters::
        X: design matrix, ndarray of shape (n_samples, n_features)
        Y: response matrix, ndarray of shape (n_samples, n_responses)
        train_idx: indices for training data for each fold if auto_split = False,
                   dictionary arranges as {n_fold: ndarray containing training indices of that fold for n_fold in range(n_folds)},
                   additionally, please include train_idx[n_folds] which returns the indices for all datapoints which is used
                   for a final fit
        val_idx: indices for CV held-out data for each fold if auto_split = False,
                 dictionary arranges as {n_fold: ndarray containing validation indices of that fold for n_fold in range(n_folds)}
        group_idx: third-party provided group for each sample for auto_split = True and split_by_group = True,
                   ndarray of shape (n_samples, )
        initial_w0: optional, initial values of intercepts, ndarray of shape (n_responses,)
        initial_w: optional, initial values of weights, ndarray of shape (n_features, n_responses)
        feature_group_mask: group mask for regularization = 'group_lasso',
                    A 2d mask array indicating groups of features for regularization, shape (num_groups, num_features).
                    Each row represents a group of features. Each column corresponds to a feature,
                    where a value of 1 indicates that the feature belongs to the group, and a value of 0 indicates it doesn't.
        verbose: print loss during fitting or not, bool

        Returns::
        self
        """
        # split data if needed
        if self.auto_split:
            train_idx, val_idx = self._cv_split(X, Y, group_idx=group_idx)
        # check train_idx and val_idx
        else:
            assert (
                train_idx is not None and val_idx is not None
            ), "Error: You must supply train_idx and val_idx if auto_split = False"
            for n_fold in range(self.n_folds):
                assert (
                    n_fold in train_idx.keys() and n_fold in val_idx.keys()
                ), "Error: Incorrect format of train_idx or val_idx. Check!"
            # add indices to all datapoint for final fit if not provided in train_idx
            if self.n_folds not in train_idx.keys():
                train_idx = {self.n_folds: np.arange(Y.shape[0])}

        self.train_idx = train_idx
        self.val_idx = val_idx

        # check number of samples in X and Y
        assert (
            X.shape[0] == Y.shape[0]
        ), "Error: Number of samples (axis 0) of X and Y not matching!"

        # reshape Y if there's only one response
        # reshape X and Y if there's only one dimension
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        # get dimension
        self.n_responses = Y.shape[1]
        self.n_features = X.shape[1]

        # generate group matrix from feature group size if regularization = group_lasso
        if self.regularization in [
            "group lasso",
            "Group Lasso",
            "groupLasso",
            "GroupLasso",
            "group_lasso",
            "Group_Lasso",
        ]:
            assert (
                feature_group_mask is not None
            ), "Error: You must provide feature_group_mask for group_lasso regularization!"
            assert (
                feature_group_mask.shape[1] == self.n_features
            ), "Error: Num of features in feature_group_mask is not equal to number of features (X.shape[1])!"
            self.feature_group_mask = jnp.array(feature_group_mask)
        else:
            self.feature_group_mask = None

        # find initial values of w0 and w
        rand_key = jax.random.key(0)
        if initial_w0 is not None:
            initial_w0 = initial_w0.reshape(1, -1)
            assert (
                initial_w0.shape[1] == self.n_responses
            ), "Error: Incorrect shape of initial_w0!"
        else:
            initial_w0 = 1e-6 * jax.random.normal(
                rand_key, shape=(self.n_responses,)
            )

        if initial_w is not None:
            assert (
                initial_w.shape[0] == self.n_features
                and initial_w.shape[1] == self.n_responses
            ), "Error: Incorrect shape of initial_w!"
        else:
            initial_w = 1e-6 * jax.random.normal(
                rand_key, shape=(self.n_features, self.n_responses)
            )

        # save target Y used for fitting (used in model selection later)
        self.Y_fit = Y

        # save design matrix X used for fitting (used in make_prediction_cv)
        self.X_fit = X

        # prelocate
        self.w_series_dict = {}
        self.all_prediction = [
            np.full(Y.shape, np.nan) for idx, _ in enumerate(self.lambda_series)
        ]
        self.all_deviance = [
            np.full((self.n_folds, self.n_responses), np.nan)
            for idx, _ in enumerate(self.lambda_series)
        ]

        self.all_frac_dev_expl = [
            np.full((self.n_folds, self.n_responses), np.nan)
            for idx, _ in enumerate(self.lambda_series)
        ]

        # fit the model
        start_time = time.time()
        for n_fold in range(
            self.n_folds + 1
        ):  # when n_fold == self.n_folds, the fit is using all datapoints
            if verbose:
                print("n_fold =", n_fold)
            start_time_fold = time.time()
            X_train = X[train_idx[n_fold], :]
            Y_train = Y[train_idx[n_fold], :]

            self.w_series_dict[n_fold] = self._fit(
                X_train, Y_train, initial_w, initial_w0
            )

            if verbose:
                print(
                    "Fitting for this fold took {:1.2f} seconds.".format(
                        time.time() - start_time_fold
                    )
                )

            # clear caches and trash
            jax.clear_caches()
            gc.collect()

            # make prediction and compute deviance on CV held-out data for the current fold (used in model selection later)
            if n_fold < self.n_folds:
                these_val_idx = val_idx[n_fold]
                X_val = X[these_val_idx, :]
                Y_val = Y[these_val_idx, :]
                for this_lambda_idx, this_w in enumerate(
                    self.w_series_dict[n_fold]
                ):
                    prediction = self.act_func(
                        jnp.matmul(X_val, this_w[1]) + this_w[0]
                    )
                    frac_dev_expl, d_model, _ = deviance(
                        np.array(prediction),
                        np.array(Y_val),
                        loss_type=self.loss_type,
                    )
                    # save prediction for CV held-out data of this fold for computing and reporting fraction deviance explained
                    self.all_prediction[this_lambda_idx][these_val_idx, :] = (
                        np.array(prediction)
                    )
                    # save deviance for CV held-out data of this fold for model selection
                    self.all_deviance[this_lambda_idx][n_fold, :] = (
                        d_model.reshape(1, -1)
                    )
                    # save fraction deviance explained for CV held-out data of this fold for model selection
                    self.all_frac_dev_expl[this_lambda_idx][n_fold, :] = (
                        frac_dev_expl.reshape(1, -1)
                    )

        if verbose:
            print(
                "Fitting took {:1.2f} seconds.".format(time.time() - start_time)
            )

        self.fitted = True

    def _fit(self, X, Y, w, w0):
        """
        <Function> Fit the model with gradient descent, used in <Method> fit
        Input parameters::
        X: design matrix, array of shape (n_samples, n_features)
        Y: response matrix, array of shape (n_samples, n_responses)
        w: intercept matrix, array of shape (1, n_responses)
        w0: weight matrix, array of shape (n_features, n_responses)

        Returns::
        w_series: fitted intercepts and weights for all lambdas,
                  list of len n_lambdas as [[w0, w] for lambda 1, [w0, w] for lambda 2, ..., etc.]
        """

        # prelocate
        w_series = []

        for i_lambda, this_lambda in enumerate(self.lambda_series):
            # set up regularizer
            if self.regularization in ["ridge", "Ridge", "L2", "l2"]:
                regularizer = nmo.regularizer.Ridge()
                solver_name = "GradientDescent"
            elif self.regularization in ["lasso", "Lasso", "L1", "l1"]:
                regularizer = nmo.regularizer.Lasso()
                solver_name = "ProximalGradient"
            elif self.regularization in [
                "group lasso",
                "Group Lasso",
                "groupLasso",
                "GroupLasso",
                "group_lasso",
                "Group_Lasso",
            ]:
                regularizer = nmo.regularizer.GroupLasso()
                solver_name = "ProximalGradient"

            # Set new solver kwargs to start with if fitting fails with the default solver kwargs
            step_size = 1.0
            decay_factor = 0.1
            max_attempts = 3
            attempt = 0
            last_exception = None  # store the last raised exception

            while attempt < max_attempts:
                try:
                    solver_kwargs = copy.deepcopy(self.solver_kwargs)
                    if attempt > 0:
                        print(
                            f"Try setting fixed stepsize to {step_size} and re-fit (Attempt {attempt})"
                        )
                        solver_kwargs["stepsize"] = step_size
                        solver_kwargs["maxiter"] = (
                            self.solver_kwargs["maxiter"] + attempt * 1000
                        )

                    model = nmo.glm.PopulationGLM(
                        observation_model=self.observation_model,
                        inverse_link_function=self.act_func,
                        regularizer=regularizer,
                        regularizer_strength=this_lambda,
                        solver_name=solver_name,
                        solver_kwargs=solver_kwargs,
                    )

                    model.fit(X, Y, init_params=(w, w0))
                    break  # fitting succeeded

                except Exception as e:
                    last_exception = e
                    print(f"Fit attempt {attempt} failed. Reducing step size.")
                    step_size *= decay_factor
                    attempt += 1

            if attempt == max_attempts:
                print(
                    "Re-fit failed after multiple attempts. Fitting unsuccessful."
                )
                raise last_exception

            # extract parameter

            w_series.append([model.intercept_, model.coef_])

        return w_series

    def _calculate_fit_quality_cv(self):
        """
        <Function> Calculate fit quality (fraction deviance explained) based on the prediction made on CV held-out data during fitting;
                   used in <Method> select_model.
                   Must be called after fitting.

        Returns::
        all_frac_dev_expl: fraction explained deviance for all responses, ndarray of shape (n_lambdas, n_response)
        all_d_model: model deviance for all responses, ndarray of shape (n_lambdas, n_response)
        all_d_null: null deviance for all responses, ndarray of shape (n_lambdas, n_response)
        """
        all_frac_dev_expl = []
        all_d_model = []

        for idx, _ in enumerate(self.lambda_series):
            prediction = self.all_prediction[idx]
            frac_dev_expl, d_model, d_null = deviance(
                prediction, self.Y_fit, loss_type=self.loss_type
            )
            all_d_model.append(d_model)
            all_frac_dev_expl.append(frac_dev_expl)
            if idx == 0:
                all_d_null = d_null
        all_frac_dev_expl = np.stack(all_frac_dev_expl, axis=0)
        all_d_model = np.stack(all_d_model, axis=0)

        return all_frac_dev_expl, all_d_model, all_d_null

    def select_model(
        self, se_fraction=1.0, min_lambda=0.0, make_fig=True, fancy_select=False
    ):
        """
        <Method> Select models using prediction and model deviance computed on CV held-out data during fitting,
                 with se_fraction that controls the tolerance of choosing models with smallest deviance vs. larger regularization.
                 After selecting the proper regularization, the attributes selected_w0 and selected_w get assigned to the
                 intercepts and weights from the re-fitted models using all datapoints with the selected regularization values.
                 This method overwrites the select_model method in GLM class. Must be called after fitting.

        Input parameters::
        X_val: design matrix for validation, array or ndarray of shape (n_samples, n_features)
        Y_val: response matrix for validation, array or ndarray of shape (n_samples, n_responses)
        se_fraction: the fraction of standard error parametrizing the tolerance of choosing models
                     with smallest deviance vs. larger regularization
                     se_fraction = 0. means selecting models with the smallest model deviance (i.e. highest explained deviance);
                     se_fraction = 1. means selecting models using the "1SE rule";
                     or set se_fraction to arbitrary positive number to control the tolerance
        min_lambda: value of minimal lambda for selection, float
        make_fig: generate plots or not, bool
        fancy_select: whether to select lambda within the SE range and with frac_dev_expl_cv >= 0, bool.
                      (if the best performing model already has frac_dev_expl_cv < 0, pick that best one)

        Returns::
        self
        """

        # Sanity check
        assert self.fitted, "Error: You have not fitted the model!"

        # computer fraction deviance explained for all lambdas based on the prediction on CV held-out data
        all_frac_dev_expl_cv, _, _ = self._calculate_fit_quality_cv()

        # compute average deviance and standard error
        avg_deviance = [np.mean(dev, axis=0) for dev in self.all_deviance]
        avg_deviance = np.stack(avg_deviance, axis=0)
        se_deviance = [
            np.std(dev, axis=0) / np.sqrt(self.n_folds)
            for dev in self.all_deviance
        ]
        se_deviance = np.stack(se_deviance, axis=0)

        # prelocate
        selected_w0 = []
        selected_w = []
        selected_lambda = []
        selected_lambda_ind = []
        selected_frac_dev_expl_cv = []
        all_min_lambda_ind = []
        all_min_lambda = []

        # find minimal lambda index
        if min_lambda > self.lambda_series.min():
            min_lambda_idx = (
                np.argwhere(self.lambda_series < min_lambda)[0][0] - 1
            )
        else:
            min_lambda_idx = self.lambda_series.shape[0]

        for idx in range(self.n_responses):
            min_deviance = np.min(avg_deviance[:, idx])
            min_dev_lambda_ind = np.argmin(avg_deviance[:, idx])
            this_se = se_deviance[min_dev_lambda_ind, idx]
            threshold = min_deviance + this_se * se_fraction

            # find the lambda index with avg deviance smaller than threshold
            this_lambda_ind = np.argwhere(avg_deviance[:, idx] <= threshold)[0][
                0
            ]

            if fancy_select:
                # if frac dev expl is already <0 for lambda with smallest deviance, select this value
                if all_frac_dev_expl_cv[min_dev_lambda_ind, idx] <= 0:
                    this_lambda_ind = min_dev_lambda_ind

                # otherwise, find the largest lambda that is within the threshold range but with >=0 frac dev expl
                else:
                    # find the lambda index for largest lambda with non-zero frac dev expl cv
                    non_zero_lambda_ind = np.argwhere(
                        all_frac_dev_expl_cv[:, idx] >= 0
                    )[0][0]

                    # pick lambda within the se range or non-zero frac dev expl cv, whichever is larger (corresponds to weaker regularization)
                    this_lambda_ind = np.max(
                        [this_lambda_ind, non_zero_lambda_ind]
                    )

            # choose between selected lambda index or min lambda index, whichever is smaller (corresponds to stronger regularization)
            this_lambda_ind = np.min([this_lambda_ind, min_lambda_idx])
            this_lambda = self.lambda_series[this_lambda_ind]

            # find fraction deviance explained for the selected lambda
            this_frac_dev = all_frac_dev_expl_cv[this_lambda_ind, idx]

            # find the corresponding weights for the lambda
            # note that w_series_dict[n_folds] returns the weights fitted with full data
            this_w0 = self.w_series_dict[self.n_folds][this_lambda_ind][0][idx]
            this_w = self.w_series_dict[self.n_folds][this_lambda_ind][1][
                :, idx
            ]

            # collect all parameters
            selected_lambda_ind.append(this_lambda_ind)
            selected_lambda.append(this_lambda)
            all_min_lambda_ind.append(min_dev_lambda_ind)
            all_min_lambda.append(self.lambda_series[min_dev_lambda_ind])
            selected_w0.append(this_w0)
            selected_w.append(this_w)
            selected_frac_dev_expl_cv.append(this_frac_dev)

        self.selected_lambda_ind = np.array(selected_lambda_ind)
        self.selected_lambda = np.array(selected_lambda)
        self.min_lambda_ind = np.array(all_min_lambda_ind)
        self.min_lambda = np.array(all_min_lambda)
        self.selected_w0 = np.stack(selected_w0, axis=0)
        self.selected_w = np.stack(selected_w, axis=1)
        self.selected_frac_dev_expl_cv = np.stack(
            selected_frac_dev_expl_cv, axis=0
        )
        self.selected = True

        if make_fig:
            self._model_selection_plot(
                all_frac_dev_expl_cv, selected_frac_dev_expl_cv
            )

    def get_coefficients_cv(self):
        """
        <Method> Get coefficients (weights and intercepts) for each fold using selected lambda values.
                 This method is used to get the coefficients for each fold after model selection.
        Input parameters::
        None
        Returns::
        w_cv: weights for selected models of each cv fold, ndarray of shape (n_folds, n_features, n_responses)
        w0_cv: intercepts for selected models of each cv fold, ndarray of shape (n_folds, n_responses)
        """
        # sanity check
        assert self.selected, "Error: You have not selected the model!"
        # loop over CV folds for making prediction on validation data
        w_cv = np.zeros((self.n_folds, self.n_features, self.n_responses))
        w0_cv = np.zeros((self.n_folds, self.n_responses))
        for n_fold in range(self.n_folds):
            # grab w and w0 for this fold
            this_w0_cv = []
            this_w_cv = []
            for n_response in range(self.n_responses):
                this_lambda_ind = self.selected_lambda_ind[n_response]
                w0 = self.w_series_dict[n_fold][this_lambda_ind][0][n_response]
                w = self.w_series_dict[n_fold][this_lambda_ind][1][
                    :, n_response
                ]
                this_w0_cv.append(w0)
                this_w_cv.append(w)

            w_cv[n_fold, :, :] = np.stack(this_w_cv, axis=1)
            w0_cv[n_fold, :] = np.stack(this_w0_cv, axis=0)
        return w_cv, w0_cv

    def make_prediction_cv(self, X_ablated=None):
        """
        <Method> Make prediction on X_ablated (or original X used for fitting) using CV fold-specific weights of selected lambdas.
                 X_ablated must be identical to X_fit (same samples) with certain features set to 0 or shuffle.
                 The prediction is made on the CV held-out data for each fold using the weights from that fold with selected lambdas.

        Input parameters::
        X_ablated: ablated design matrix identical to the original X used for fitting (same samples) with certain features set to 0 or shuffle,
                   ndarray of shape (n_samples_fit, n_features). If None, the original X_fit is used to make cv prediction on fit data.

        Returns::
        pred: prediction made on X_ablated using CV fold-specific weights of selected lambdas,
              ndarray of shape (n_samples_fit, n_responses)
        """

        # sanity check
        assert self.selected, "Error: You have not selected the model!"
        if X_ablated is None:
            this_X = self.X_fit.copy()
        else:
            assert (
                X_ablated.shape == self.X_fit.shape
            ), "Error: Shape of X_ablated is not identical to that of X used for fitting!"
            this_X = X_ablated.copy()

        # prelocate
        pred = np.empty((this_X.shape[0], self.n_responses))

        # loop over CV folds for making prediction on validation data
        for n_fold in range(self.n_folds):
            # grab validation indices for this fold
            these_val_frames = self.val_idx[n_fold]

            # grab w and w0 for this fold
            this_w0_cv = []
            this_w_cv = []
            for n_response in range(self.n_responses):
                this_lambda_ind = self.selected_lambda_ind[n_response]
                w0 = self.w_series_dict[n_fold][this_lambda_ind][0][n_response]
                w = self.w_series_dict[n_fold][this_lambda_ind][1][
                    :, n_response
                ]
                this_w0_cv.append(w0)
                this_w_cv.append(w)

            this_w_cv = np.stack(this_w_cv, axis=1)
            this_w0_cv = np.stack(this_w0_cv, axis=0)

            # make predictions on validation frames of this fold
            this_pred = self.act_func(
                jnp.matmul(this_X[these_val_frames, :], this_w_cv) + this_w0_cv
            )

            pred[these_val_frames, :] = np.array(this_pred)

        return pred

    def _model_selection_plot(
        self, all_frac_dev_expl, selected_frac_dev_expl, bin_width=0.05
    ):
        """
        <Function> Make plots for model selection, used in <Method> select_model
        Input parameters::
        all_frac_dev_expl: fraction explained deviance for all responses, ndarray of shape (n_lambdas, n_response)
        selected_frac_dev_expl: fraction deviance explained for selected models

        Returns::
        self
        """

        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        axes[0].plot(
            np.log10(self.lambda_series),
            all_frac_dev_expl,
            color="k",
            linewidth=0.5,
        )
        axes[0].set_ylim((-1, 1))
        axes[0].set_xlabel("log_lambda")
        axes[0].set_ylabel("Fraction deviance explained")
        axes[0].set_title("Fraction deviance explained vs. lambda")

        axes[1].hist(selected_frac_dev_expl, bins=np.arange(-1, 1, bin_width))
        axes[1].set_xlabel("Fraction deviance explained")
        axes[1].set_ylabel("Count")
        axes[1].set_title(
            "Distribution of fraction deviance explained \n for selected models on validation data"
        )
        plt.tight_layout()

    def predict(self, X):
        """<Method> Make prediction using selected model weights. Must be called after model selection.
        Input parameters::
        X: design matrix for test, ndarray of shape (n_samples, n_features)

        Returns::
        Y_pred: predicted response matrix, ndarray of shape (n_samples, n_responses)
        """
        assert (
            self.selected
        ), "Error: You have to perform model selection with validation data first before making prediction!"

        # reshape X if there's only one dimension
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        assert (
            X.shape[1] == self.n_features
        ), "Error: Incorrect number of features (axis 1) in X!"

        # make prediction
        Y_pred = self.act_func(
            jnp.matmul(X, self.selected_w) + self.selected_w0
        )
        return np.array(Y_pred)

    def evaluate(self, X_test, Y_test, make_fig=True):
        """<Method> Evaluate selected model with test data using selected weights.
                    Must be called after model selection.
        Input parameters::
        X_test: design matrix for test, array of shape (n_samples, n_features)
        Y_test: response matrix for test, array of shape (n_samples, n_responses)
        make_fig: generate plots or not, bool

        Returns::
        frac_dev_expl: fraction deviance explained for all responses, ndarray of shape (n_responses,)
        dev_model: model deviance for all responses, ndarray of shape (n_responses,)
        dev_null: null deviance for all responses, ndarray of shape (n_responses,)
        dev_expl: deviance explained for all responses (null deviance - model deviance), ndarray of shape (n_responses,)
        """

        # reshape X_test and Y_test if there's only one dimension
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        if Y_test.ndim == 1:
            Y_test = Y_test.reshape(-1, 1)

        # sanity check
        assert (
            self.selected
        ), "Error: You have to perform model selection with validation data first before evaluating!"
        assert (
            X_test.shape[0] == Y_test.shape[0]
        ), "Error: Number of datapoints (axis 0) of X_test and Y_test not matching!"
        assert (
            X_test.shape[1] == self.n_features
        ), "Error: Incorrect number of features (axis 1) in X_test!"
        assert (
            Y_test.shape[1] == self.n_responses
        ), "Error: Incorrect number of response (axis 1) in Y_test!"

        # select best model for each source
        frac_dev_expl = []
        dev_model = []
        dev_null = []
        dev_expl = []

        # make prediction on test set and calculate fraction deviance explained, model deviance, and null deviance
        prediction = self.predict(X_test)
        for idx in range(self.n_responses):
            best_frac_deviance, best_d_model, best_d_null = deviance(
                prediction[:, idx], Y_test[:, idx], loss_type=self.loss_type
            )
            best_dev_expl = best_d_null - best_d_model

            frac_dev_expl.append(best_frac_deviance)
            dev_model.append(best_d_model)
            dev_null.append(best_d_null)
            dev_expl.append(best_dev_expl)

        if make_fig:
            # set negative frac dev expl value to zeros (for printing only)
            frac_dev_expl_clipped = np.array(frac_dev_expl)
            frac_dev_expl_clipped[frac_dev_expl_clipped < 0] = 0
            print(
                "Fraction deviance explained: mean = {:1.4f}, median = {:1.4f}".format(
                    np.mean(frac_dev_expl_clipped),
                    np.median(np.array(frac_dev_expl)),
                )
            )

            # plot CDF for fraction deviance explained / scatter of  deviance explained vs null deviance
            density, bins = np.histogram(
                frac_dev_expl, bins=np.arange(0, 1, 0.01), density=True
            )
            this_ecdf = np.cumsum(density)

            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
            axes[0].plot(bins[1:], this_ecdf)
            axes[0].set_xlabel("Fraction deviance explained")
            axes[0].set_ylabel("Cumulative density")
            axes[0].set_title("CDF for fraction deviance explained")

            axes[1].plot(dev_null, dev_expl, ".", markersize=3)
            axes[1].plot(
                np.linspace(0, np.max(dev_null), 100),
                np.linspace(0, np.max(dev_null), 100),
                linestyle="--",
                linewidth=1,
                color=(0.5, 0.5, 0.5),
            )
            axes[1].set_xlim([0, np.max(dev_null)])
            axes[1].set_ylim([0, np.max(dev_expl)])
            axes[1].set_xlabel("Deviance for null model")
            axes[1].set_ylabel("Deviance explained")
            axes[1].set_title("Deviance explained vs. null deviance")
            plt.tight_layout()

        frac_dev_expl = np.array(frac_dev_expl)
        dev_model = np.array(dev_model)
        dev_null = np.array(dev_null)
        dev_expl = np.array(dev_expl)

        return frac_dev_expl, dev_model, dev_null, dev_expl


def stable(x, eps=1e-33):
    """
    Add a tiny positive constant to input value to stablize it when taking log (avoid log(0))
    """
    return x + eps


def pointwise_deviance(y_true, y_pred, loss_type="poisson"):
    """
    Compute pointwise deviance for data with given loss type
    Input parameters::
    y_true: true values, ndarray
    y_pred: predicted values, ndarray
    loss_type: {'gaussian', 'poisson', 'binominal','gamma'}, default = 'poisson'

    Returns::
    dev_pt: pointwise deviance value, ndarray of shape of y_true and y_pred
    """

    assert (
        y_true.shape == y_true.shape
    ), "Shapes of y_true and y_pred don't match!"
    if loss_type == "poisson":
        dev_pt = 2.0 * (
            y_true * (np.log(stable(y_true)) - np.log(stable(y_pred)))
            + y_pred
            - y_true
        )
    elif loss_type == "gamma":
        dev_pt = 2.0 * (
            np.log(stable(y_pred))
            - np.log(stable(y_true))
            + y_true / stable(y_pred)
            - 1
        )
    elif loss_type == "gaussian":
        dev_pt = (y_true - y_pred) ** 2
    elif loss_type == "binominal":
        dev_pt = 2.0 * (
            -y_true * np.log(stable(y_pred))
            - (1.0 - y_true) * np.log(stable(1.0 - y_pred))
            + y_true * np.log(stable(y_true))
            + (1.0 - y_true) * np.log(stable(1.0 - y_true))
        )
    return dev_pt


def pointwise_null_deviance(y, loss_type="poisson"):
    """
    Compute pointwise null deviance for data with given loss type
    Input parameters::
    y: input data, ndarray
    loss_type: {'gaussian', 'poisson', 'binominal','gamma'}, default = 'poisson'

    Returns::
    null_dev_pt: pointwise null deviance value, ndarray of shape of y
    """
    mean_y = np.mean(y, axis=0)
    null_dev_pt = pointwise_deviance(y, mean_y, loss_type=loss_type)
    return null_dev_pt


def null_deviance(y, loss_type="poisson"):
    """
    Compute null deviance for data with given loss type, average over n_samples for each response
    Input parameters::
    y: input data, ndarray of shape (n_samples, n_responses)
    loss_type: {'gaussian', 'poisson', 'binominal','gamma'}, default = 'poisson'

    Returns::
    null_dev: average null deviance for each response, ndarray of shape of (n_responses,)
    """
    mean_y = np.mean(y, axis=0)
    null_dev = np.sum(
        pointwise_deviance(y, mean_y, loss_type=loss_type), axis=0
    )
    return null_dev


def deviance(y_pred, y_true, loss_type="poisson"):
    """
    Compute fraction deviance explained, model deviance and null deviance for data with given loss type,
    averaged over n_samples for each response
    Input parameters::
    y_pred: predicted values, ndarray of shape (n_samples, n_responses)
    y_true: true values, ndarray of shape (n_samples, n_responses)
    loss_type: {'gaussian', 'poisson', 'binominal','gamma'}, default = 'poisson'

    Returns::
    frac_dev_expl: average fraction deviance explained for each response, ndarray of shape of (n_responses,)
    d_model: average model deviance for each response, ndarray of shape of (n_responses,)
    d_null: average null deviance for each response, ndarray of shape of (n_responses,)
    """

    mean_y = np.mean(y_true, axis=0)
    d_null = np.sum(
        pointwise_deviance(y_true, mean_y, loss_type=loss_type), axis=0
    )
    d_model = np.sum(
        pointwise_deviance(y_true, y_pred, loss_type=loss_type), axis=0
    )
    frac_dev_expl = 1.0 - d_model / stable(d_null)

    if isinstance(
        frac_dev_expl, type(y_true)
    ):  # if dev is still an ndarray (skip if is a single number)
        # If mean_y == 0, we get 0 for model and null deviance, i.e. 0/0 in the deviance fraction.
        if isinstance(frac_dev_expl, np.ndarray):
            frac_dev_expl[mean_y == 0] = 0
        else:
            frac_dev_expl = frac_dev_expl.at[mean_y == 0].set(0)

    return frac_dev_expl, d_model, d_null


def make_prediction(X, w, w0, activation="exp"):
    """
    Make GLM prediction
    Input parameters::
    X: design matrix, ndarray of shape (n_samples, n_features)
    w: weight matrix, ndarray of shape (n_features, n_responses)
    w0: intercept matrix, ndarray of shape (1, n_responses)
    activation: {'linear', 'exp', 'sigmoid', 'relu', 'softplus'}, default = 'exp'

    Returns::
    prediction: model prediction, ndarray of shape (n_samples, n_responses)
    """
    if activation == "exp":
        prediction = np.exp(w0 + np.matmul(X, w))
    elif activation == "relu":
        prediction = np.maximum((w0 + np.matmul(X, w)), 0)
    elif activation == "softplus":
        prediction = np.log(
            stable(np.exp(w0 + np.matmul(X, w)) + 1.0)
        )  # take softplus = log(exp(features) + 1
    elif activation == "linear":
        prediction = w0 + np.matmul(X, w)
    elif activation == "sigmoid":
        prediction = 1.0 / (1.0 + np.exp(-w0 - np.matmul(X, w)))
    return prediction