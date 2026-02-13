"""
Python wrapper for C++ adamCore class methods.

This module provides backward-compatible wrappers that:
1. Maintain the old function-based API while using the new adamCore class
2. Convert C++ FitResult/ForecastResult objects to dictionaries
3. Handle type conversions between Python and C++

The functions in this module are low-level wrappers around the C++ implementation
of ADAM (Augmented Dynamic Adaptive Model). They handle the interface between
Python and C++ by ensuring proper data types, memory layouts (Fortran-contiguous
arrays), and result conversions.

For user-facing functionality, use the ADAM class from smooth.adam_general.core.adam
instead of calling these functions directly.
"""

from typing import Dict

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
    Fit ADAM model using C++ adamCore implementation.

    This is a low-level wrapper around the C++ fitting routine. It converts
    Python arrays to the proper format, calls the C++ fitter, and returns
    the results as a dictionary.

    Parameters
    ----------
    matrixVt : NDArray[np.float64]
        Initial state matrix (nComponents x nObs+1). Contains initial values
        for all state components (level, trend, seasonal, ARIMA states).
    matrixWt : NDArray[np.float64]
        Measurement matrix (nComponents x nObs). Maps state to observation.
    matrixF : NDArray[np.float64]
        Transition matrix (nComponents x nComponents). Defines state evolution.
    vectorG : NDArray[np.float64]
        Persistence vector (nComponents,). Smoothing parameters.
    lags : NDArray[np.uint64]
        Vector of lags for seasonal components and ARIMA.
    indexLookupTable : NDArray[np.uint64]
        Lookup table for indexing lagged states.
    profilesRecent : NDArray[np.float64]
        Recent states for ARIMA components.
    E : str
        Error type: 'A' (Additive) or 'M' (Multiplicative).
    T : str
        Trend type: 'N' (None), 'A' (Additive), 'Ad' (Additive damped),
        'M' (Multiplicative), 'Md' (Multiplicative damped).
    S : str
        Seasonal type: 'N' (None), 'A' (Additive), 'M' (Multiplicative).
    nNonSeasonal : int
        Number of non-seasonal components.
    nSeasonal : int
        Number of seasonal components.
    nArima : int
        Number of ARIMA components.
    nXreg : int
        Number of exogenous regressors.
    constant : bool
        Whether to include a constant term.
    vectorYt : NDArray[np.float64]
        Observed time series values (nObs,).
    vectorOt : NDArray[np.float64]
        Occurrence vector (nObs,). 1 for observed, 0 for missing.
    backcast : bool
        Whether to use backcasting for initialization.
    nIterations : int
        Number of refining iterations for state updates.
    refineHead : bool
        Whether to refine the initial state.
    adamETS : bool
        Whether this is a pure ETS model (vs ARIMA or mixed).

    Returns
    -------
    Dict[str, NDArray[np.float64]]
        Dictionary with keys:
        - 'matVt': Updated state matrix (nComponents x nObs+1)
        - 'yFitted': Fitted values (nObs,)
        - 'errors': Residuals (nObs,)
        - 'profile': Likelihood profile for optimization

    Notes
    -----
    - All input arrays are converted to Fortran-contiguous layout for C++ efficiency
    - The C++ implementation uses BLAS/LAPACK for matrix operations
    - This function is called internally by the ADAM class estimator

    See Also
    --------
    adam_forecaster : Generate forecasts from fitted model
    adam_simulator : Simulate trajectories from model
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
    Generate forecasts from fitted ADAM model using C++ implementation.

    This is a low-level wrapper around the C++ forecasting routine. It takes
    the fitted model parameters and generates point forecasts for the specified
    horizon.

    Parameters
    ----------
    matrixWt : NDArray[np.float64]
        Measurement matrix (nComponents x horizon). Maps state to observation.
        Extended to forecast horizon.
    matrixF : NDArray[np.float64]
        Transition matrix (nComponents x nComponents). Defines state evolution.
    lags : NDArray[np.uint64]
        Vector of lags for seasonal components and ARIMA.
    indexLookupTable : NDArray[np.uint64]
        Lookup table for indexing lagged states.
    profilesRecent : NDArray[np.float64]
        Recent states from fitted model, used as initial conditions.
    E : str
        Error type: 'A' (Additive) or 'M' (Multiplicative).
    T : str
        Trend type: 'N' (None), 'A' (Additive), 'Ad' (Additive damped),
        'M' (Multiplicative), 'Md' (Multiplicative damped).
    S : str
        Seasonal type: 'N' (None), 'A' (Additive), 'M' (Multiplicative).
    nNonSeasonal : int
        Number of non-seasonal components.
    nSeasonal : int
        Number of seasonal components.
    nArima : int
        Number of ARIMA components.
    nXreg : int
        Number of exogenous regressors.
    constant : bool
        Whether model includes a constant term.
    horizon : int
        Number of periods to forecast.

    Returns
    -------
    NDArray[np.float64]
        Forecast values array of shape (horizon,).

    Notes
    -----
    - Forecasts are point predictions (means) only; intervals computed separately
    - All input arrays are converted to Fortran-contiguous layout for C++
    - The state space is projected forward using the transition matrix
    - This function is called internally by the ADAM class predict method

    See Also
    --------
    adam_fitter : Fit model parameters
    adam_simulator : Simulate forecast trajectories with uncertainty
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
    Simulate time series trajectories from ADAM model using C++ implementation.

    This is a low-level wrapper around the C++ simulation routine. It generates
    simulated time series by propagating random errors through the state space
    model structure.

    Parameters
    ----------
    matrixErrors : NDArray[np.float64]
        Matrix of random errors (nPaths x nObs). Each row is an error trajectory.
    matrixOt : NDArray[np.float64]
        Occurrence matrix (nPaths x nObs). For intermittent demand, 1=demand occurs.
    arrayVt : NDArray[np.float64]
        Initial state vector for all components.
    matrixWt : NDArray[np.float64]
        Measurement matrix (nComponents x nObs). Maps state to observation.
    arrayF : NDArray[np.float64]
        Transition matrix (nComponents x nComponents). Defines state evolution.
    matrixG : NDArray[np.float64]
        Persistence matrix (nComponents x nObs). Smoothing parameters over time.
    lags : NDArray[np.uint64]
        Vector of lags for seasonal components and ARIMA.
    indexLookupTable : NDArray[np.uint64]
        Lookup table for indexing lagged states.
    profilesRecent : NDArray[np.float64]
        Recent historical states for initialization.
    E : str
        Error type: 'A' (Additive) or 'M' (Multiplicative).
        Determines how errors affect observations and states.
    T : str
        Trend type: 'N' (None), 'A' (Additive), 'Ad' (Additive damped),
        'M' (Multiplicative), 'Md' (Multiplicative damped).
    S : str
        Seasonal type: 'N' (None), 'A' (Additive), 'M' (Multiplicative).
    nNonSeasonal : int
        Number of non-seasonal components (level, trend).
    nSeasonal : int
        Number of seasonal components.
    nArima : int
        Number of ARIMA components (AR, MA terms).
    nXreg : int
        Number of exogenous regressors.
    constant : bool
        Whether model includes a constant term.

    Returns
    -------
    Dict[str, NDArray[np.float64]]
        Dictionary with keys:
        - 'arrayVt': Simulated states array (nComponents x nObs)
        - 'matrixYt': Simulated observations matrix (nPaths x nObs)

    Notes
    -----
    - Used for Monte Carlo simulation of forecast paths
    - Enables computation of prediction intervals via simulation
    - All input arrays are converted to Fortran-contiguous layout for C++
    - Each simulated path uses independent error draws
    - This function is called internally for scenario analysis and uncertainty
      quantification

    See Also
    --------
    adam_fitter : Fit model parameters
    adam_forecaster : Generate point forecasts

    Examples
    --------
    Typically used internally, but can generate custom scenarios::

        # After fitting model, simulate 1000 paths
        errors = np.random.normal(0, residual_std, (1000, horizon))
        results = adam_simulator(errors, ...)
        simulated_data = results['matrixYt']
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
