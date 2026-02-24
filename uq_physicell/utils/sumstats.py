import pcdl
import numpy as np
import pandas as pd
from shutil import rmtree
from typing import Union
import re
    
def summ_func_FinalPopLiveDead(outputPath:str,summaryFile:Union[str,None], dic_params:dict, SampleID:int, ReplicateID:int) -> Union[pd.DataFrame,None]:
    """
    Final population of live and dead cells
    
    Parameters:
    - outputPath: str -> Path to the PhysiCell output directory.
    - summaryFile: Union[str, None] -> File to store the summary (optional).
    - dic_params: dict -> Dictionary of simulation parameters.
    - SampleID: int -> Unique identifier for the sample.
    - ReplicateID: int -> Unique identifier for the replicate.
    
    Returns:
    - pd.DataFrame or None -> DataFrame with the computed QoIs or None if saved to a file.
    """
    # read the last file
    mcds = pcdl.TimeStep('final.xml',outputPath, microenv=False, graph=False, settingxml=None, verbose=False)
    # dataframe of cells
    df_cell = mcds.get_cell_df() 
    # population stats live and dead cells
    live_cells = len(df_cell[ (df_cell['dead'] == False) ] )
    dead_cells = len(df_cell[ (df_cell['dead'] == True) ] )
    # dataframe structure
    data = {'time': mcds.get_time(), 'sampleID': SampleID, 'replicateID': ReplicateID, 'live_cells': live_cells, 'dead_cells': dead_cells, 'run_time_sec': mcds.get_runtime()}
    data_conc = {**data,**dic_params} # concatenate output data and parameters
    df = pd.DataFrame([data_conc])
    # remove replicate output folder
    rmtree( outputPath )
    if (summaryFile): 
        df.to_csv(summaryFile, sep='\t', encoding='utf-8')
        return None
    else: return df

# Population over time of live and dead cells
def summ_func_TimeSeriesPopLiveDead(outputPath:str,summaryFile:Union[str,None], dic_params:dict, SampleID:int, ReplicateID:int) -> Union[pd.DataFrame,None]:
    """
    Population over time of live and dead cells

    Parameters:
    - outputPath: str -> Path to the PhysiCell output directory.
    - summaryFile: Union[str, None] -> File to store the summary (optional).
    - dic_params: dict -> Dictionary of simulation parameters.
    - SampleID: int -> Unique identifier for the sample.
    - ReplicateID: int -> Unique identifier for the replicate.
    
    Returns:
    - pd.DataFrame or None -> DataFrame with the computed QoIs or None if saved to a file.
    """

    mcds_ts = pcdl.TimeSeries(outputPath, microenv=False, graph=False, settingxml=None, verbose=False)
    for mcds in mcds_ts.get_mcds_list():
        df_cell = mcds.get_cell_df() 
        live_cells = len(df_cell[ (df_cell['dead'] == False) ] )
        dead_cells = len(df_cell[ (df_cell['dead'] == True) ] )
        data = {'time': mcds.get_time(), 'sampleID': SampleID, 'replicateID': ReplicateID, 'live_cells': live_cells, 'dead_cells': dead_cells, 'run_time_sec': mcds.get_runtime()}
        data_conc = {**data,**dic_params} # concatenate output data and parameters
        if ( mcds.get_time() == 0 ): df = pd.DataFrame([data_conc]) # create the dataframe
        else: df.loc[len(df)] = data_conc # append the dictionary to the dataframe
    # remove replicate output folder
    rmtree( outputPath )
    if (summaryFile): 
        df.to_csv(summaryFile, sep='\t', encoding='utf-8')
        return None
    else: return df

def check_functions_need_microenv(qoi_funcs):
    """
    Check if any of the QoI functions require microenvironment data (df_subs).
    
    Parameters:
    - qoi_funcs: dict -> Dictionary of QoI functions
    
    Returns:
    - bool: True if any function needs microenvironment data
    """
    # If no functions provided, return True (default to loading microenvironment)
    if not qoi_funcs:
        return True
    
    # If any function needs microenvironment data, return True
    for func in qoi_funcs.values():
        # Check if any parameter name indicates substrate/concentration data
        param_lower = func.__param_name__.lower()
        if 'subs' in param_lower or 'conc' in param_lower or 'micro' in param_lower:
                return True
    # If none need it,
    return False

def safe_call_qoi_function(func, mcds=None, list_mcds=None ):
    """
    Safely call a QoI function with the appropriate dataframe based on parameter inspection.
    
    Parameters:
    - func: The QoI function to call
    - mcds: pcdl.TimeStep or None -> The mcds object for single snapshot
    - list_mcds: pcdl.TimeSeries or None -> The mcds time series object for multiple snapshots
    
    Returns:
    - Result of the QoI function
    """
    # Check if function has our custom parameter name attribute (from string creation)
    param_name = func.__param_name__
    if param_name in ['df_cell', 'df'] and mcds is not None: # Function expects cell dataframe
        return func(mcds.get_cell_df())
    elif param_name in ['df_subs'] and mcds is not None: # Function expects substrate dataframe
        return func(mcds.get_conc_df())
    elif param_name in ['mcds'] and mcds is not None: # Function expects the mcds object
        return func(mcds)
    elif param_name in ['mcds_ts'] and list_mcds is not None: # Function expects the mcds time series object
        if mcds == list_mcds[-1]: # Ensure we only compute once per time series (last mcds passed)
            return func(list_mcds)
        else: return None # Skip computation for other snapshots
    else:
        raise ValueError(f"Unknown parameter name '{param_name}' for QoI function.")
    
def create_named_function_from_string(func_str: str, qoi_name: str) -> callable:
    """
    Dynamically creates a named function from a string.
    Parameters:
    - func_str: The string representation of the function.
    - qoi_name: The name of the function to be created.
    Return:
    - The created function with preserved parameter inspection capability.
    """
    # Extract arg name from lambda (e.g., "lambda df_subs:" -> "df_subs")
    arg_match = re.search(r'lambda\s+(\w+):', func_str)
    if arg_match:
        arg_name = arg_match.group(1)
    else:
        arg_name = 'mcds'
    
    # Create a restricted namespace with necessary imports and no built-in functions
    namespace = {'pd': pd, 'np': np, 'len': len, 'sum': sum, 'map': map, '__builtins__': {}}
    
    # Evaluate the lambda function once at creation time
    try:
        compiled_func = eval(func_str, namespace)
    except Exception as e:
        raise ValueError(f"Error evaluating QoI function string '{func_str}': {e}")
    
    # Create a wrapper function that calls the pre-compiled lambda
    def wrapper(*args, **kwargs):
        return compiled_func(*args, **kwargs)
    
    # Store the original parameter name as an attribute for inspection
    wrapper.__param_name__ = arg_name
    wrapper.__name__ = qoi_name
    
    return wrapper

def recreate_qoi_functions(qoi_functions: dict) -> dict:
    """
    Recreate QoI functions from their string representations.
    Parameters:
    - qoi_functions: Dictionary of QoI functions (keys as names, values as strings)
    Return:
    - Dictionary of recreated QoI functions (keys as names, values as callables)
    """
    recreated_qoi_funcs = {}
    for qoi_name, qoi_value in qoi_functions.items():
        try:
            recreated_qoi_funcs[qoi_name] = create_named_function_from_string(qoi_value, qoi_name)
        except Exception as e:
            raise ValueError(f"Error recreating QoI function '{qoi_name}': {e}")
    return recreated_qoi_funcs
    
def summary_function(outputPath:str, summaryFile:Union[str, None], dic_params:dict, SampleID:int, ReplicateID:int, qoi_functions:dict,  mode:str='time_series', RemoveFolder:bool=True, drop_columns:Union[list, None]=None,) -> Union[pd.DataFrame, None]:
    """
    Generic summary function for creating custom QoIs (Quantities of Interest) based on df_cell elements.

    Parameters:
    - outputPath: str -> Path to the PhysiCell output directory.
    - summaryFile: Union[str, None] -> File to store the summary (optional).
    - dic_params: dict -> Dictionary of simulation parameters.
    - SampleID: int -> Unique identifier for the sample.
    - ReplicateID: int -> Unique identifier for the replicate.
    - qoi_functions: dict -> Dictionary of QoI functions with keys as QoI names and values as functions/lambdas.
    - mode: str -> Mode of operation: 'last_snapshot', 'time_series', or 'summary'.
    - RemoveFolder: bool -> Whether to remove the output folder after processing.
    - drop_columns: list -> List of columns to drop from the DataFrame.

    Returns:
    - pd.DataFrame or None -> DataFrame with the computed QoIs or None if saved to a file.
    """
    try:
        if mode == 'last_snapshot':
            # Check if any function needs microenvironment data
            needs_microenv = check_functions_need_microenv(qoi_functions)
            
            # Load the last snapshot
            mcds = pcdl.TimeStep('final.xml', outputPath, microenv=needs_microenv, graph=False, settingxml=None, verbose=False)
            if qoi_functions is None:
                # Optional: Remove replicate output folder
                if (RemoveFolder): rmtree(outputPath)
                # Entire mcds is returned if drop_columns is empty
                if not drop_columns:
                    return [mcds]
                else:
                    df_cell = mcds.get_cell_df()  # Ensure df_cell is initialized
                    df_cell.drop(columns=drop_columns, inplace=True, errors='ignore')
                return [df_cell]
                    
            else:
                # Compute QoIs using safe function calling
                qoi_data = {name: safe_call_qoi_function(func, mcds=mcds) for name, func in qoi_functions.items()}
                data = {
                    'time': mcds.get_time(),
                    'sampleID': SampleID,
                    'replicateID': ReplicateID,
                    **qoi_data
                }
                data_conc = {**data, **dic_params}
                df = pd.DataFrame([data_conc])

        elif mode == 'time_series':
            # Check if any function needs microenvironment data
            needs_microenv = check_functions_need_microenv(qoi_functions)
            # Load the time series
            mcds_ts = pcdl.TimeSeries(outputPath, microenv=needs_microenv, graph=False, settingxml=None, verbose=False)
            list_mcds=mcds_ts.get_mcds_list()
            #  All data is stored as a list of mcds
            if qoi_functions is None:
                # Optional: Remove replicate output folder
                if (RemoveFolder): rmtree(outputPath)
                # Entire list of mcds is returned if drop_columns is empty
                if not drop_columns:
                    return list_mcds
                else:
                    df_list = []
                    for mcds in list_mcds:
                        df_cell = mcds.get_cell_df()
                        df_cell.drop(columns=drop_columns, inplace=True, errors='ignore')
                        df_list.append(df_cell)
                    return df_list
            else:
                df_list = []
                for mcds in list_mcds:
                    try:
                        qoi_data = {}
                        for name, func in qoi_functions.items():
                            func_result = safe_call_qoi_function(func, mcds=mcds, list_mcds=list_mcds)
                            if func_result is not None:
                                qoi_data[name] = func_result
                        if not qoi_data: continue  # Skip if no QoI data was computed
                        # print(f"Computed QoIs at time {mcds.get_time()}: {qoi_data}")
                    except Exception as e:
                        raise RuntimeError(f"Error computing QoIs in summary_function: {e}")
                    
                    data = {
                        'time': mcds.get_time(),
                        'sampleID': SampleID,
                        'replicateID': ReplicateID,
                        **qoi_data
                    }
                    data_conc = {**data, **dic_params}
                    df_list.append(data_conc)
                df = pd.DataFrame(df_list)
                

        elif mode == 'summary':
            # Check if any function needs microenvironment data
            needs_microenv = check_functions_need_microenv(qoi_functions)
            
            # Load the time series and summarize across snapshots
            mcds_ts = pcdl.TimeSeries(outputPath, microenv=needs_microenv, graph=False, settingxml=None, verbose=False)
            summary_data = {name: [] for name in qoi_functions.keys()}
            for mcds in mcds_ts.get_mcds_list():
                for name, func in qoi_functions.items():
                    summary_data[name].append(safe_call_qoi_function(func, mcds=mcds))
            summarized_qois = {name: sum(values) / len(values) for name, values in summary_data.items()}
            data = {
                'sampleID': SampleID,
                'replicateID': ReplicateID,
                **summarized_qois
            }
            data_conc = {**data, **dic_params}
            df = pd.DataFrame([data_conc])

        else:
            raise ValueError(f"Invalid mode: {mode}. Supported modes are 'last_snapshot', 'time_series', and 'summary'.")
    except FileNotFoundError as e:
        raise RuntimeError(f"Error: Required file not found in {outputPath}. {e}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while processing QoIs: {e}")

    # Optional: Remove replicate output folder
    if (RemoveFolder): rmtree(outputPath)

    # Save to file or return DataFrame
    if summaryFile:
        df.to_csv(summaryFile, sep='\t', encoding='utf-8')
        return None
    else:
        return df

def compute_persistent_homology(df:pd.DataFrame, Plot=False) -> tuple:
    """
    Compute persistent homology vectorization using muspan.
    (source: https://docs.muspan.co.uk/latest/_collections/topology/Topology%203%20-%20persistence%20vectorisation.html)
    
    Parameters:
    - df_cells: DataFrame -> DataFrame containing cell data with 'position_x', 'position_y', and 'cell_type' columns.
    - Plot: bool -> Whether to plot the persistence diagram (optional).
    
    Returns:
    - tuple -> (pd.Series with vectorized persistent homology features, figure or None)
    """
    import matplotlib.pyplot as plt
    try:
        import muspan
    except ImportError:
        raise ImportError("muspan library is required for computing persistent homology. Please install it via 'pip install muspan'.")

    # Extract cell positions
    points = df[['position_x', 'position_y']].to_numpy()
    labels = df['cell_type'].to_numpy()
    
    # Create a muspan domain and add points
    domain = muspan.domain('Position Data')
    domain.add_points(points, 'Cell Positions')
    domain.add_labels('Celltype', labels)

    # Query to select cells of types 'A', 'B', ... in each domain
    q_cell_types = muspan.query.query(domain, ('label', 'Celltype'), 'in', df['cell_type'].unique().tolist())

    # Compute Vietoris-Rips filtrations
    feature_persistence = muspan.topology.vietoris_rips_filtration(domain, population=q_cell_types, max_dimension=1)

    # Plot persistence diagram (optional)
    figure = None
    if Plot:
        fig, axes = plt.subplots(figsize=(15, 6), nrows=1, ncols=2)
        # Plot domain with cell types
        muspan.visualise.visualise(domain, 'Celltype', ax=axes[0], add_cbar=False, marker_size=2.5, objects_to_plot=q_cell_types)
        # Plot persistence diagram
        muspan.visualise.persistence_diagram(feature_persistence, ax=axes[1])
        figure = [fig, axes]
        

    # Vectorise the persistence homology diagram for the domain using statistical method
    vectorised_ph,name_of_features = muspan.topology.vectorise_persistence(feature_persistence, method='statistics')
    return pd.Series(vectorised_ph, index=name_of_features), figure

def compute_relational_ph( df: pd.DataFrame, landmark_type: str, witness_type: str, max_dim: int = 1, mode: str = "distance", ax = None) -> tuple:
    """
    Relational Persistent Homology using Dowker/Witness idea.
    A→B (A as vertices, B as witnesses) describes how B are arranged around A geometry.

    Parameters
    ----------
    df : DataFrame with columns ['position_x','position_y','cell_type']
    landmark_type : cell type to use as landmarks (A)
    witness_type : cell type to use as witnesses (B)
    max_dim : PH dimension (0 or 1)
    mode: 'distance' or 'count'
    ax : matplotlib axis for plotting  (optional)

    Returns
    -------
    tuple -> (pd.Series with vectorized persistent homology features, persistence diagram)
    """
    from scipy.spatial.distance import cdist
    from scipy.spatial import Delaunay
    import gudhi, muspan
    import matplotlib.pyplot as plt

    # Extract A = Landmarks, B = Witnesses
    A_points = df[df["cell_type"] == landmark_type][["position_x","position_y"]].to_numpy()
    B_points = df[df["cell_type"] == witness_type][["position_x","position_y"]].to_numpy()
    if len(A_points) == 0 or len(B_points) == 0:
        raise ValueError("Both landmark_type and witness_type must be present.")
    # Pairwise distances B×A
    D = cdist(B_points, A_points)

    # Build candidate simplex list from landmarks
    if len(A_points) < 3:
        raise ValueError("At least 3 landmark points are required for relational PH.")
    # Use Delaunay to match Python repo behavior
    tri = Delaunay(A_points)
    # vertices = all points
    vertices = list(range(len(A_points)))
    # edges from Delaunay
    edges = set()
    faces = set()
    for simplex in tri.simplices:
        simplex = list(simplex)
        # all edges
        for i in range(3):
            for j in range(i+1, 3):
                edges.add(tuple(sorted([simplex[i], simplex[j]])))
        # triangle
        faces.add(tuple(sorted(simplex)))
    edges = list(edges)
    faces = list(faces)

    # Compute filtration values
    def dowker_distance(simplex):
        """Distance-based Dowker: min_w max_a d(a,w)."""
        return np.min(np.max(D[:, simplex], axis=1))
    
    def dowker_count(simplex):
        """Count-based Dowker: number of witnesses covering all simplex vertices."""
        return -np.sum(np.all(D[:, simplex] <= np.max(D[:, simplex], axis=0), axis=1))
    
    # Compute filtration values based on the distance
    fil_func = dowker_distance if mode == "distance" else dowker_count
    f_vertex = np.array([fil_func([i]) for i in vertices])
    f_edge   = np.array([fil_func(list(e)) for e in edges])
    if max_dim >= 2:
        f_face   = np.array([fil_func(list(f)) for f in faces])

    # Build GUDHI SimplexTree
    st = gudhi.SimplexTree()
    # vertices
    for i, fv in enumerate(f_vertex):
        st.insert([i], filtration=float(fv))
    # edges
    for (e, fv) in zip(edges, f_edge):
        st.insert(list(e), filtration=float(fv))
    # faces
    if max_dim >= 2:
        for (f, fv) in zip(faces, f_face):
            st.insert(list(f), filtration=float(fv))
    st.initialize_filtration()

    # Compute persistence
    diag = st.persistence()

    # Convert GUDHI persistence output to muspan-style dict {'dgms': [array_dim0, array_dim1, ...]}
    dgms = []
    for d in range(max_dim + 1):
        intervals = [pair for dim, pair in diag if dim == d]
        if len(intervals) == 0:
            dgms.append(np.zeros((0, 2)))
        else:
            dgms.append(np.array(intervals))
    feature_persistence = {'dgms': dgms}

    # Vectorize diagram using muspan's statistics
    vec, names = muspan.topology.vectorise_persistence(feature_persistence, method="statistics")
    vec = pd.Series(vec, index=names)

    # Plot
    if ax is not None:
        axes = gudhi.plot_persistence_diagram(diag, axes=ax)

    return vec, diag