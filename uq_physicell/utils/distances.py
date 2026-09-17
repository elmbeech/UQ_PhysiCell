import numpy as np

def relative_rmse(sim, obs, key: str) -> float:
    """
    Relative RMSE between simulated (sim) and observed (obs) values of one QoI.
    Wrap it per key to build a per-QoI distance function, e.g.::

        def distance_live_cells(sim, obs):
            return relative_rmse(sim, obs, "live_cell_count")

    Returns np.inf rather than raising or treating a missing/malformed
    simulation output as a perfect (0.0) fit, which would otherwise silently
    bias a calibration toward whichever parameter regions fail most often.
    Handles two cases where the two sides genuinely cannot be compared:

    - ``sim`` is None: the simulation for this particle failed.
    - ``sim[key]`` is a DataFrame with ``['time', key]`` columns rather than
      the plain 1-D array a live simulation produces for this same key --
      the shape a value takes after being serialized to and reconstructed
      from storage. The column is pulled out here so both shapes end up
      comparable.

    Args:
        sim (dict or None): Simulated summary statistics, keyed by QoI name
            (e.g. ``sim["live_cell_count"]``). None if the simulation failed.
        obs (dict): Observed data, keyed the same way as ``sim``.
        key (str): Which QoI (dict key) to compare.

    Returns:
        float: Relative RMSE over the time points where both sim and obs are
            finite (a QoI measured only at an endpoint, like a cumulative
            death count, is NaN everywhere else), or np.inf when sim is None,
            the key/shape don't line up, or there is no overlapping data.
    """
    if sim is None:
        return np.inf
    try:
        o = np.asarray(obs[key], dtype=float)
        sim_val = sim[key]
        if hasattr(sim_val, "columns"):    # reconstructed-from-storage DataFrame, not a live array
            sim_val = sim_val[key]
        s = np.asarray(sim_val, dtype=float)
    except (KeyError, TypeError):          # QoI missing/malformed in an otherwise "successful" result
        return np.inf
    if s.shape != o.shape:                 # run stopped early / malformed output
        return np.inf
    mask = np.isfinite(o) & np.isfinite(s) # e.g. an endpoint-only measurement
    if not mask.any():
        return np.inf
    o, s = o[mask], s[mask]
    return float(np.sqrt(np.mean(((o - s) / np.maximum(np.abs(o), 1.0)) ** 2)))

def SumSquaredDifferences(dic_model_data:dict, dic_obs_data:dict)-> float:
    """
    Compute the sum of squared differences between simulation outputs and observational data.
    Args:
        dic_model_data (dict): Dictionary containing model data with keys "time" and "value".
            None when the simulation for this particle failed (e.g. PhysiCell crashed) --
            see CalibrationContext._run_physicell_model.
        dic_obs_data (dict): Dictionary containing observational data with keys "time" and "value".
    Returns:
        float: The sum of squared differences between the model data and observational data,
            or np.inf if dic_model_data is None so pyABC rejects the particle instead of
            crashing the worker on `None["time"]`.
    """
    if dic_model_data is None:
        return np.inf
    indices_model = np.where(np.isin(dic_model_data["time"], dic_obs_data["time"]))[0]
    indices_obsData = np.where(np.isin(dic_obs_data["time"], dic_model_data["time"]))[0]
    if len(indices_model) == 0 or len(indices_obsData) == 0:
        raise ValueError("No matching time points found between model data and observational data.")
    diff = dic_model_data["value"][indices_model] - dic_obs_data["value"][indices_obsData]
    return np.sum(diff ** 2)

def Manhattan(dic_model_data:dict, dic_obs_data:dict)-> float:
    """
    Compute the Manhattan distance (L1 norm) between simulation outputs and observational data.
    Args:
        dic_model_data (dict): Dictionary containing model data with keys "time" and "value".
            None when the simulation for this particle failed (e.g. PhysiCell crashed) --
            see CalibrationContext._run_physicell_model.
        dic_obs_data (dict): Dictionary containing observational data with keys "time" and "value".
    Returns:
        float: The Manhattan distance between the model data and observational data,
            or np.inf if dic_model_data is None so pyABC rejects the particle instead of
            crashing the worker on `None["time"]`.
    """
    if dic_model_data is None:
        return np.inf
    indices_model = np.where(np.isin(dic_model_data["time"], dic_obs_data["time"]))[0]
    indices_obsData = np.where(np.isin(dic_obs_data["time"], dic_model_data["time"]))[0]
    if len(indices_model) == 0 or len(indices_obsData) == 0:
        raise ValueError("No matching time points found between model data and observational data.")
    diff = dic_model_data["value"][indices_model] - dic_obs_data["value"][indices_obsData]
    return np.sum(np.abs(diff))

def Chebyshev(dic_model_data:dict, dic_obs_data:dict)-> float:
    """
    Compute the Chebyshev distance (L∞ norm) between simulation outputs and observational data.
    Args:
        dic_model_data (dict): Dictionary containing model data with keys "time" and "value".
            None when the simulation for this particle failed (e.g. PhysiCell crashed) --
            see CalibrationContext._run_physicell_model.
        dic_obs_data (dict): Dictionary containing observational data with keys "time" and "value".
    Returns:
        float: The Chebyshev distance between the model data and observational data,
            or np.inf if dic_model_data is None so pyABC rejects the particle instead of
            crashing the worker on `None["time"]`.
    """
    if dic_model_data is None:
        return np.inf
    indices_model = np.where(np.isin(dic_model_data["time"], dic_obs_data["time"]))[0]
    indices_obsData = np.where(np.isin(dic_obs_data["time"], dic_model_data["time"]))[0]
    if len(indices_model) == 0 or len(indices_obsData) == 0:
        raise ValueError("No matching time points found between model data and observational data.")
    diff = dic_model_data["value"][indices_model] - dic_obs_data["value"][indices_obsData]
    return np.max(np.abs(diff))
