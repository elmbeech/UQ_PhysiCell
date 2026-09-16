import numpy as np

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
