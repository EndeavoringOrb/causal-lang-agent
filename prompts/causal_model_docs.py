class CausalModel:
    """Main class for storing the causal model state."""

    def __init__(
        self,
        data,
        treatment,
        outcome,
        graph=None,
        common_causes=None,
        instruments=None,
        effect_modifiers=None,
        estimand_type="nonparametric-ate",
        proceed_when_unidentifiable=False,
        missing_nodes_as_confounders=False,
        identify_vars=False,
        **kwargs,
    ):
        """Initialize data and create a causal graph instance.

        Assigns treatment and outcome variables.
        Also checks and finds the common causes and instruments for treatment
        and outcome.

        At least one of graph, common_causes or instruments must be provided. If
        none of these variables are provided, then learn_graph() can be used later.

        :param data: a pandas dataframe containing treatment, outcome and other variables.
        :param treatment: name of the treatment variable
        :param outcome: name of the outcome variable
        :param graph: path to DOT file containing a DAG or a string containing a DAG specification in DOT format
        :param common_causes: names of common causes of treatment and _outcome. Only used when graph is None.
        :param instruments: names of instrumental variables for the effect of treatment on outcome. Only used when graph is None.
        :param effect_modifiers: names of variables that can modify the treatment effect. If not provided, then the causal graph is used to find the effect modifiers. Estimators will return multiple different estimates based on each value of effect_modifiers.
        :param estimand_type: the type of estimand requested (currently only "nonparametric-ate" is supported). In the future, may support other specific parametric forms of identification.
        :param proceed_when_unidentifiable: does the identification proceed by ignoring potential unobserved confounders. Binary flag.
        :param missing_nodes_as_confounders: Binary flag indicating whether variables in the dataframe that are not included in the causal graph, should be  automatically included as confounder nodes.
        :param identify_vars: Variable deciding whether to compute common causes, instruments and effect modifiers while initializing the class. identify_vars should be set to False when user is providing common_causes, instruments or effect modifiers on their own(otherwise the identify_vars code can override the user provided values). Also it does not make sense if no graph is given.
        :returns: an instance of CausalModel class

        """

    def identify_effect(
        self,
        estimand_type=None,
        method_name="default",
        proceed_when_unidentifiable=None,
        optimize_backdoor=False,
    ):
        """Identify the causal effect to be estimated, using properties of the causal graph.

        :param method_name: Method name for identification algorithm. ("id-algorithm" or "default")
        :param proceed_when_unidentifiable: Binary flag indicating whether identification should proceed in the presence of (potential) unobserved confounders.
        :returns: a probability expression (estimand) for the causal effect if identified, else NULL

        """

    def estimate_effect(
        self,
        identified_estimand,
        method_name=None,
        control_value=0,
        treatment_value=1,
        test_significance=None,
        evaluate_effect_strength=False,
        confidence_intervals=False,
        target_units="ate",
        effect_modifiers=None,
        fit_estimator=True,
        method_params=None,
    ):
        """Estimate the identified causal effect.

        Currently requires an explicit method name to be specified. Method names follow the convention of identification method followed by the specific estimation method: "[backdoor/iv/frontdoor].estimation_method_name". For a list of supported methods, check out the :doc:`User Guide </user_guide/causal_tasks/estimating_causal_effects/index>`. Here are some examples.

            * Propensity Score Matching: "backdoor.propensity_score_matching"
            * Propensity Score Stratification: "backdoor.propensity_score_stratification"
            * Propensity Score-based Inverse Weighting: "backdoor.propensity_score_weighting"
            * Linear Regression: "backdoor.linear_regression"
            * Generalized Linear Models (e.g., logistic regression): "backdoor.generalized_linear_model"
            * Instrumental Variables: "iv.instrumental_variable"
            * Regression Discontinuity: "iv.regression_discontinuity"
            * Two Stage Regression: "frontdoor.two_stage_regression"

        In addition, you can directly call any of the EconML estimation methods. The convention is "[backdoor/iv].econml.path-to-estimator-class". For example, for the double machine learning estimator ("DML" class) that is located inside "dml" module of EconML, you can use the method name, "backdoor.econml.dml.DML". See :doc:`this demo notebook </example_notebooks/dowhy-conditional-treatment-effects>`.


        :param identified_estimand: a probability expression
            that represents the effect to be estimated. Output of
            CausalModel.identify_effect method
        :param method_name: name of the estimation method to be used.
        :param control_value: Value of the treatment in the control group, for effect estimation.  If treatment is multi-variate, this can be a list.
        :param treatment_value: Value of the treatment in the treated group, for effect estimation. If treatment is multi-variate, this can be a list.
        :param test_significance: Binary flag on whether to additionally do a statistical signficance test for the estimate.
        :param evaluate_effect_strength: (Experimental) Binary flag on whether to estimate the relative strength of the treatment's effect. This measure can be used to compare different treatments for the same outcome (by running this method with different treatments sequentially).
        :param confidence_intervals: (Experimental) Binary flag indicating whether confidence intervals should be computed.
        :param target_units: (Experimental) The units for which the treatment effect should be estimated. This can be of three types. (1) a string for common specifications of target units (namely, "ate", "att" and "atc"), (2) a lambda function that can be used as an index for the data (pandas DataFrame), or (3) a new DataFrame that contains values of the effect_modifiers and effect will be estimated only for this new data.
        :param effect_modifiers: Names of effect modifier variables can be (optionally) specified here too, since they do not affect identification. If None, the effect_modifiers from the CausalModel are used.
        :param fit_estimator: Boolean flag on whether to fit the estimator.
            Setting it to False is useful to estimate the effect on new data using a previously fitted estimator.
        :param method_params: Dictionary containing any method-specific parameters. These are passed directly to the estimating method. See the docs for each estimation method for allowed method-specific params.
        :returns: An instance of the CausalEstimate class, containing the causal effect estimate
            and other method-dependent information

        """

    def refute_estimate(
        self, estimand, estimate, method_name=None, show_progress_bar=False, **kwargs
    ):
        """Refute an estimated causal effect.

        If method_name is provided, uses the provided method. In the future, we may support automatic selection of suitable refutation tests. Following refutation methods are supported.
            * Adding a randomly-generated confounder: "random_common_cause"
            * Adding a confounder that is associated with both treatment and outcome: "add_unobserved_common_cause"
            * Replacing the treatment with a placebo (random) variable): "placebo_treatment_refuter"
            * Removing a random subset of the data: "data_subset_refuter"

        :param estimand: target estimand, an instance of the IdentifiedEstimand class (typically, the output of identify_effect)
        :param estimate: estimate to be refuted, an instance of the CausalEstimate class (typically, the output of estimate_effect)
        :param method_name: name of the refutation method
        :param show_progress_bar: Boolean flag on whether to show a progress bar
        :param kwargs:  (optional) additional arguments that are passed directly to the refutation method. Can specify a random seed here to ensure reproducible results ('random_seed' parameter). For method-specific parameters, consult the documentation for the specific method. All refutation methods are in the causal_refuters subpackage.

        :returns: an instance of the RefuteResult class

        """
