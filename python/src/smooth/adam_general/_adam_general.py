"""
Python wrapper for C++ adamCore class methods.

This module provides backward-compatible wrappers that:
1. Maintain the old function-based API while using the new adamCore class
2. Convert C++ FitResult/ForecastResult objects to dictionaries
3. Handle type conversions between Python and C++

The wrappers interface with the C++ implementation of ADAM (Augmented Dynamic
Adaptive Model) for efficient state-space model fitting, forecasting, and
simulation. All functions handle proper memory layout (Fortran-contiguous arrays)
and type conversions required by the C++ backend.

Functions
---------
adam_fitter : Fit ADAM model to observed data
adam_forecaster : Generate forecasts from fitted ADAM model
adam_simulator : Simulate data from ADAM model
"""

from typing import Any, Dict

import numpy as np
from numpy.typing import NDArray

from smooth.adam_general import _adamCore


def adam_fitter(
    matrixVt: NDArray[np.float64],
    matrixWt: NDArray[np.float64],
    matrixF: NDArray[np.float64],
    vectorG: NDArray[np.float64],
    lags: NDArray[np.uint64],
    indexLookupTable: NDArray[np.uint64],
    profilesRecent: NDArray[np.float64],
    E: str,
    T: str,
    S: str,
    nNonSeasonal: int,
    nSeasonal: int,
    nArima: int,
    nXreg: int,
    constant: bool,
    vectorYt: NDArray[np.float64],
    vectorOt: NDArray[np.float64],
    backcast: bool,
    nIterations: int,
    refineHead: bool,
    adamETS: bool,
) -> Dict[str, NDArray[np.float64]]:
    """
    Fit ADAM model to observed time series data.

    This function is a backward-compatible wrapper for the adamCore.fit() C++ method.
    It performs state-space model fitting using the ADAM framework, which combines
    Error-Trend-Seasonal (ETS) and ARIMA components in a unified single source of
    error model.

    The ADAM model is represented in state-space form:

        y_t = o_t * (w(v_{t-l}) + h(x_t, a_{t-1}) + r(v_{t-l}) * ε_t)
        v_t = f(v_{t-l}, a_{t-1}) + g(v_{t-l}, a_{t-1}, x_t) * ε_t

    where v_t is the state vector, y_t is the observed value, and ε_t is the error term.

    Parameters
    ----------
    matrixVt : NDArray[np.float64]
        State matrix (nComponents x nObs). Contains level, trend, seasonal, and
        ARIMA state components. Will be updated in-place with fitted states.
        Shape: (nComponents, nObs)
    matrixWt : NDArray[np.float64]
        Measurement matrix (nObs x nComponents). Maps state vector to observations.
        Shape: (nObs, nComponents)
    matrixF : NDArray[np.float64]
        Transition matrix (nComponents x nComponents). Defines state evolution.
        Shape: (nComponents, nComponents)
    vectorG : NDArray[np.float64]
        Persistence (smoothing) vector (nComponents,). Controls how errors
        affect state updates. Contains smoothing parameters α, β, γ, etc.
        Shape: (nComponents,)
    lags : NDArray[np.uint64]
        Lag structure for seasonal components. First element is typically 1
        (for non-seasonal components), followed by seasonal periods (e.g., [1, 12]
        for monthly data with annual seasonality).
        Shape: (nLags,)
    indexLookupTable : NDArray[np.uint64]
        Index lookup table for handling multiple seasonal lags efficiently.
        Maps current time to appropriate lagged state indices.
        Shape: (maxLag, nObs)
    profilesRecent : NDArray[np.float64]
        Recent state profiles for multi-seasonal models. Stores recent states
        for all lags to enable efficient state updates.
        Shape: (nComponents, maxLag)
    E : str
        Error type specification: 'A' (additive) or 'M' (multiplicative).
    T : str
        Trend type: 'N' (none), 'A' (additive), 'Ad' (additive damped),
        'M' (multiplicative), 'Md' (multiplicative damped).
    S : str
        Seasonality type: 'N' (none), 'A' (additive), 'M' (multiplicative).
    nNonSeasonal : int
        Number of non-seasonal components (level, trend, ARIMA states).
    nSeasonal : int
        Number of seasonal components.
    nArima : int
        Number of ARIMA states (AR + MA orders across all lags).
    nXreg : int
        Number of exogenous regressors.
    constant : bool
        Whether the model includes a constant/intercept term.
    vectorYt : NDArray[np.float64]
        Observed time series values (nObs,).
        Shape: (nObs,)
    vectorOt : NDArray[np.float64]
        Occurrence indicator vector (nObs,). Binary vector where 1 indicates
        non-zero observation (for intermittent demand). All 1s for non-intermittent data.
        Shape: (nObs,)
    backcast : bool
        Whether to use backcasting for initial state estimation. If True,
        fits model backwards from start to refine initial states.
    nIterations : int
        Number of iterations for backcasting refinement.
    refineHead : bool
        Whether to refine initial states (head of time series).
    adamETS : bool
        Whether this is a pure ETS model (True) or includes ARIMA/other
        components (False).

    Returns
    -------
    Dict[str, NDArray[np.float64]]
        Dictionary containing fitted model results:

        - 'matVt' : NDArray[np.float64]
            Fitted state matrix (nComponents x nObs). Contains the complete
            state trajectory over the sample period.
        - 'yFitted' : NDArray[np.float64]
            Fitted values (nObs,). One-step-ahead in-sample predictions.
        - 'errors' : NDArray[np.float64]
            Residuals (nObs,). Difference between actual and fitted values.
        - 'profile' : NDArray[np.float64]
            Recent state profile (nComponents x maxLag). Final state information
            for forecasting.

    Notes
    -----
    - All input arrays are converted to Fortran-contiguous float64/uint64 format
      for C++ compatibility.
    - The function modifies matrixVt in-place but returns a copy in the result dict.
    - For multiplicative models, ensure vectorYt contains only positive values.
    - The C++ implementation uses Armadillo library for efficient matrix operations.

    See Also
    --------
    adam_forecaster : Generate forecasts from fitted model
    adam_simulator : Simulate data from ADAM model

    Examples
    --------
    >>> import numpy as np
    >>> # Setup simple model matrices for N=100 observations
    >>> nObs = 100
    >>> matrixVt = np.zeros((2, nObs))  # Level + Trend
    >>> matrixWt = np.ones((nObs, 2))
    >>> matrixF = np.eye(2)
    >>> vectorG = np.array([0.1, 0.05])
    >>> lags = np.array([1], dtype=np.uint64)
    >>> # ... (setup other parameters)
    >>> result = adam_fitter(matrixVt, matrixWt, matrixF, vectorG, lags, ...)
    >>> fitted_values = result['yFitted']
    >>> residuals = result['errors']
    """
    # Convert types to ensure C++ compatibility
    lags = np.asarray(lags, dtype=np.uint64).ravel()

    # Create adamCore instance
    adam_core = _adamCore.adamCore(
        lags=lags,
        E=E,
        T=T,
        S=S,
        nNonSeasonal=int(nNonSeasonal),
        nSeasonal=int(nSeasonal),
        nETS=int(nNonSeasonal + nSeasonal),
        nArima=int(nArima),
        nXreg=int(nXreg),
        constant=bool(constant),
        adamETS=bool(adamETS),
    )

    # Ensure matrices are F-contiguous
    matrixVt = np.asfortranarray(matrixVt, dtype=np.float64)
    matrixWt = np.asfortranarray(matrixWt, dtype=np.float64)
    matrixF = np.asfortranarray(matrixF, dtype=np.float64)
    vectorG = np.asfortranarray(vectorG, dtype=np.float64).ravel()
    indexLookupTable = np.asfortranarray(indexLookupTable, dtype=np.uint64)
    profilesRecent = np.asfortranarray(profilesRecent, dtype=np.float64)
    vectorYt = np.asfortranarray(vectorYt, dtype=np.float64).ravel()
    vectorOt = np.asfortranarray(vectorOt, dtype=np.float64).ravel()

    # Call C++ fit method
    result = adam_core.fit(
        matrixVt=matrixVt,
        matrixWt=matrixWt,
        matrixF=matrixF,
        vectorG=vectorG,
        indexLookupTable=indexLookupTable,
        profilesRecent=profilesRecent,
        vectorYt=vectorYt,
        vectorOt=vectorOt,
        backcast=bool(backcast),
        nIterations=int(nIterations),
        refineHead=bool(refineHead),
    )

    # Convert C++ FitResult to dict for backward compatibility
    return {
        "matVt": np.array(result.states),
        "yFitted": np.array(result.fitted),
        "errors": np.array(result.errors),
        "profile": np.array(result.profile),
    }


def adam_forecaster(
    matrixWt: NDArray[np.float64],
    matrixF: NDArray[np.float64],
    lags: NDArray[np.uint64],
    indexLookupTable: NDArray[np.uint64],
    profilesRecent: NDArray[np.float64],
    E: str,
    T: str,
    S: str,
    nNonSeasonal: int,
    nSeasonal: int,
    nArima: int,
    nXreg: int,
    constant: bool,
    horizon: int,
) -> NDArray[np.float64]:
    """
    Generate point forecasts from fitted ADAM model.

    This function is a backward-compatible wrapper for the adamCore.forecast() C++ method.
    It projects the state-space model forward to generate multi-step-ahead forecasts
    using the fitted model structure and parameters.

    The forecast recursively applies the state transition equation:

        v_{t+h} = f(v_{t+h-1}, a_t)
        ŷ_{t+h} = w(v_{t+h})

    for h = 1, 2, ..., horizon, where v_t is the state vector at time t.

    Parameters
    ----------
    matrixWt : NDArray[np.float64]
        Measurement matrix (horizon x nComponents). Maps state vector to forecasted
        observations. Typically constant but can vary for models with exogenous
        variables.
        Shape: (horizon, nComponents)
    matrixF : NDArray[np.float64]
        Transition matrix (nComponents x nComponents). Defines how states evolve
        during the forecast period. Must be the same as used in fitting.
        Shape: (nComponents, nComponents)
    lags : NDArray[np.uint64]
        Lag structure for seasonal components. Must match the lags used in fitting.
        First element typically 1, followed by seasonal periods (e.g., [1, 12]).
        Shape: (nLags,)
    indexLookupTable : NDArray[np.uint64]
        Index lookup table for multi-seasonal models. Maps forecast periods to
        appropriate lagged state indices.
        Shape: (maxLag, horizon)
    profilesRecent : NDArray[np.float64]
        Recent state profile from fitting (nComponents x maxLag). Contains the
        final states from the fitted model, which serve as initial conditions
        for forecasting. This should be the 'profile' output from adam_fitter.
        Shape: (nComponents, maxLag)
    E : str
        Error type specification: 'A' (additive) or 'M' (multiplicative).
        Must match the fitted model.
    T : str
        Trend type: 'N' (none), 'A' (additive), 'Ad' (additive damped),
        'M' (multiplicative), 'Md' (multiplicative damped).
        Must match the fitted model.
    S : str
        Seasonality type: 'N' (none), 'A' (additive), 'M' (multiplicative).
        Must match the fitted model.
    nNonSeasonal : int
        Number of non-seasonal components. Must match the fitted model.
    nSeasonal : int
        Number of seasonal components. Must match the fitted model.
    nArima : int
        Number of ARIMA states. Must match the fitted model.
    nXreg : int
        Number of exogenous regressors. If > 0, matrixWt should include
        future values of exogenous variables.
    constant : bool
        Whether the model includes a constant term. Must match the fitted model.
    horizon : int
        Number of periods to forecast ahead. Must be > 0.

    Returns
    -------
    NDArray[np.float64]
        Point forecasts for the next 'horizon' periods. Array of shape (horizon,)
        containing the expected value of the time series at each future period.

    Notes
    -----
    - All input arrays are converted to Fortran-contiguous float64/uint64 format
      for C++ compatibility.
    - Point forecasts represent conditional expectations: E[y_{t+h} | I_t] where
      I_t is information available at time t.
    - For multiplicative models, forecasts are constrained to be positive.
    - Damped trends gradually flatten as horizon increases.
    - Seasonal patterns repeat according to the specified lags.
    - This function generates point forecasts only. For prediction intervals,
      additional simulation is needed (see adam_simulator).

    See Also
    --------
    adam_fitter : Fit ADAM model to data
    adam_simulator : Simulate forecast trajectories for prediction intervals

    Examples
    --------
    >>> import numpy as np
    >>> # After fitting a model, use the profile for forecasting
    >>> horizon = 12
    >>> matrixWt = np.ones((horizon, 2))  # Simple measurement
    >>> matrixF = np.eye(2)  # Identity transition
    >>> lags = np.array([1], dtype=np.uint64)
    >>> # ... (setup other parameters from fitted model)
    >>> forecasts = adam_forecaster(matrixWt, matrixF, lags, ...)
    >>> # forecasts now contains 12-step-ahead point predictions
    """
    # Convert types
    lags = np.asarray(lags, dtype=np.uint64).ravel()

    # Create adamCore instance
    adam_core = _adamCore.adamCore(
        lags=lags,
        E=E,
        T=T,
        S=S,
        nNonSeasonal=int(nNonSeasonal),
        nSeasonal=int(nSeasonal),
        nETS=int(nNonSeasonal + nSeasonal),
        nArima=int(nArima),
        nXreg=int(nXreg),
        constant=bool(constant),
        adamETS=False,
    )

    # Ensure matrices are F-contiguous
    matrixWt = np.asfortranarray(matrixWt, dtype=np.float64)
    matrixF = np.asfortranarray(matrixF, dtype=np.float64)
    indexLookupTable = np.asfortranarray(indexLookupTable, dtype=np.uint64)
    profilesRecent = np.asfortranarray(profilesRecent, dtype=np.float64)

    # Call C++ forecast method
    result = adam_core.forecast(
        matrixWt=matrixWt,
        matrixF=matrixF,
        indexLookupTable=indexLookupTable,
        profilesRecent=profilesRecent,
        horizon=int(horizon),
    )

    # Return forecast array directly
    return np.array(result.forecast)


def adam_simulator(
    matrixErrors: NDArray[np.float64],
    matrixOt: NDArray[np.float64],
    arrayVt: NDArray[np.float64],
    matrixWt: NDArray[np.float64],
    arrayF: NDArray[np.float64],
    matrixG: NDArray[np.float64],
    lags: NDArray[np.uint64],
    indexLookupTable: NDArray[np.uint64],
    profilesRecent: NDArray[np.float64],
    E: str,
    T: str,
    S: str,
    nNonSeasonal: int,
    nSeasonal: int,
    nArima: int,
    nXreg: int,
    constant: bool,
) -> Dict[str, NDArray[np.float64]]:
    """
    Simulate time series data from ADAM model.

    This function is a backward-compatible wrapper for the adamCore.simulate() C++ method.
    It generates synthetic time series by propagating random errors through the
    state-space model structure. This is useful for:

    - Computing prediction intervals (multiple simulations)
    - Model diagnostics and validation
    - Understanding model behavior under different error scenarios
    - Monte Carlo analysis

    The simulation recursively applies the ADAM state-space equations:

        y_t = o_t * (w(v_{t-l}) + r(v_{t-l}) * ε_t)
        v_t = f(v_{t-l}) + g(v_{t-l}) * ε_t

    where ε_t are the provided error terms.

    Parameters
    ----------
    matrixErrors : NDArray[np.float64]
        Error matrix (nPaths x nObs). Each row contains a sequence of random
        errors for one simulation path. Errors should be drawn from an appropriate
        distribution (e.g., standard normal for additive models).
        Shape: (nPaths, nObs)
    matrixOt : NDArray[np.float64]
        Occurrence matrix (nPaths x nObs). Binary indicators for intermittent
        demand models. Use all 1s for non-intermittent data. Each row corresponds
        to one simulation path.
        Shape: (nPaths, nObs)
    arrayVt : NDArray[np.float64]
        Initial state vector (nComponents,). Starting values for the state
        components (level, trend, seasonal, ARIMA states). Typically obtained
        from the final state of a fitted model.
        Shape: (nComponents,)
    matrixWt : NDArray[np.float64]
        Measurement matrix (nObs x nComponents). Maps state vector to observations.
        Shape: (nObs, nComponents)
    arrayF : NDArray[np.float64]
        Transition matrix (nComponents x nComponents). Defines state evolution.
        Shape: (nComponents, nComponents)
    matrixG : NDArray[np.float64]
        Persistence matrix (nComponents x nObs). Smoothing parameters that
        control how errors affect state updates. Can be time-varying.
        Shape: (nComponents, nObs)
    lags : NDArray[np.uint64]
        Lag structure for seasonal components. First element typically 1,
        followed by seasonal periods (e.g., [1, 12] for monthly data).
        Shape: (nLags,)
    indexLookupTable : NDArray[np.uint64]
        Index lookup table for multi-seasonal models. Maps current time to
        appropriate lagged state indices.
        Shape: (maxLag, nObs)
    profilesRecent : NDArray[np.float64]
        Recent state profile (nComponents x maxLag). Contains recent states
        for all lags to initialize the simulation properly.
        Shape: (nComponents, maxLag)
    E : str
        Error type specification: 'A' (additive) or 'M' (multiplicative).
    T : str
        Trend type: 'N' (none), 'A' (additive), 'Ad' (additive damped),
        'M' (multiplicative), 'Md' (multiplicative damped).
    S : str
        Seasonality type: 'N' (none), 'A' (additive), 'M' (multiplicative).
    nNonSeasonal : int
        Number of non-seasonal components (level, trend, ARIMA states).
    nSeasonal : int
        Number of seasonal components.
    nArima : int
        Number of ARIMA states (sum of AR and MA orders).
    nXreg : int
        Number of exogenous regressors.
    constant : bool
        Whether the model includes a constant/intercept term.

    Returns
    -------
    Dict[str, NDArray[np.float64]]
        Dictionary containing simulation results:

        - 'arrayVt' : NDArray[np.float64]
            Simulated state trajectories (nComponents x nObs x nPaths).
            Contains the complete evolution of all state components across
            all simulation paths.
        - 'matrixYt' : NDArray[np.float64]
            Simulated observation matrix (nPaths x nObs). Each row contains
            one simulated time series trajectory.

    Notes
    -----
    - All input arrays are converted to Fortran-contiguous float64/uint64 format
      for C++ compatibility.
    - Error terms should be appropriately scaled for the model's error type:

      * Additive errors: typically standard normal N(0,1)
      * Multiplicative errors: ensure they don't cause negative values

    - For prediction intervals, run multiple simulations (large nPaths) and
      compute quantiles of matrixYt.
    - The function is deterministic given fixed error terms, enabling
      reproducible results.

    See Also
    --------
    adam_fitter : Fit ADAM model to data
    adam_forecaster : Generate point forecasts

    Examples
    --------
    >>> import numpy as np
    >>> # Simulate 1000 paths of length 100
    >>> nPaths, nObs = 1000, 100
    >>> matrixErrors = np.random.normal(0, 1, (nPaths, nObs))
    >>> matrixOt = np.ones((nPaths, nObs))  # Non-intermittent
    >>> arrayVt = np.array([100.0, 0.0])  # Initial level and trend
    >>> # ... (setup other parameters)
    >>> result = adam_simulator(matrixErrors, matrixOt, arrayVt, ...)
    >>> simulated_data = result['matrixYt']
    >>> # Compute 95% prediction intervals
    >>> lower_bound = np.percentile(simulated_data, 2.5, axis=0)
    >>> upper_bound = np.percentile(simulated_data, 97.5, axis=0)
    """
    # Convert types
    lags = np.asarray(lags, dtype=np.uint64).ravel()

    # Create adamCore instance
    adam_core = _adamCore.adamCore(
        lags=lags,
        E=E,
        T=T,
        S=S,
        nNonSeasonal=int(nNonSeasonal),
        nSeasonal=int(nSeasonal),
        nETS=int(nNonSeasonal + nSeasonal),
        nArima=int(nArima),
        nXreg=int(nXreg),
        constant=bool(constant),
        adamETS=False,
    )

    # Ensure matrices are F-contiguous
    matrixErrors = np.asfortranarray(matrixErrors, dtype=np.float64)
    matrixOt = np.asfortranarray(matrixOt, dtype=np.float64)
    arrayVt = np.asfortranarray(arrayVt, dtype=np.float64)
    matrixWt = np.asfortranarray(matrixWt, dtype=np.float64)
    arrayF = np.asfortranarray(arrayF, dtype=np.float64)
    matrixG = np.asfortranarray(matrixG, dtype=np.float64)
    indexLookupTable = np.asfortranarray(indexLookupTable, dtype=np.uint64)
    profilesRecent = np.asfortranarray(profilesRecent, dtype=np.float64)

    # Call C++ simulate method
    result = adam_core.simulate(
        matrixErrors=matrixErrors,
        matrixOt=matrixOt,
        arrayVt=arrayVt,
        matrixWt=matrixWt,
        arrayF=arrayF,
        matrixG=matrixG,
        indexLookupTable=indexLookupTable,
        profilesRecent=profilesRecent,
        E=E,
    )

    # Convert to dict
    return {
        "arrayVt": np.array(result.states),
        "matrixYt": np.array(result.data),
    }
