
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .abc_context import CalibrationContext
from uq_physicell import __version__ as uq_physicell_version
from pcdl import __version__ as pcdl_version
from pyabc import __version__ as pyabc_version

def insert_adaptive_weights_db(db_file, dict_distances, dict_adaptive_weights):
    import sqlite3
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    # Create AdaptiveDistance Table
    cursor.execute(f"""CREATE TABLE IF NOT EXISTS AdaptiveDistance (
                Population INTEGER PRIMARY KEY,
                {', '.join([f'{distance} DOUBLE' for distance in dict_distances.keys()])})""")
    conn.commit()
    conn.close()
    # Insert the Data
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    for pop_id, weights in dict_adaptive_weights.items():
        placeholders = ', '.join(['?'] * (len(dict_distances)))
        cursor.execute(f"""INSERT OR IGNORE INTO AdaptiveDistance (Population, {', '.join(dict_distances.keys())}) VALUES (?, {placeholders})""", (pop_id, *weights))
    conn.commit()
    conn.close()

def _prior_summary(prior) -> dict:
    """Best-effort JSON-serializable summary of a pyABC prior Distribution."""
    try:
        out = {}
        for pname in prior.get_parameter_names():
            rv = prior[pname]
            out[pname] = {
                "distribution": getattr(rv, "name", None),
                "args": list(getattr(rv, "args", []) or []),
                "kwargs": dict(getattr(rv, "kwargs", {}) or {}),
            }
        return out
    except Exception:
        return {"repr": repr(prior)}


def _model_config_fingerprint(spec) -> dict:
    """Best-effort PhysiCell effective-config fingerprint for one model spec.

    Returns ``{}`` (every hash resolves to None) if the model cannot be instantiated.
    """
    try:
        from uq_physicell import PhysiCell_Model
        pc_model = PhysiCell_Model(spec.model_config["ini_path"], spec.model_config["struc_name"])
        return pc_model.build_effective_config_fingerprint()
    except Exception:
        return {}


def insert_models_db(db_file: str, abc_context: "CalibrationContext"):
    """Store the candidate ABC-SMC model specs in a ``Models`` table.

    One row per model in ``abc_context.models`` (a single row for a plain
    single-model calibration). Keeps pyABC's own schema untouched. The per-model
    effective-config fingerprint hashes let a reader tell candidates that differ in
    their PhysiCell configuration (``struc_name``, XML, rules) apart from ones that
    only differ by prior / fixed parameters.
    """
    import json
    import sqlite3
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS Models (
                    ModelIndex INTEGER PRIMARY KEY,
                    Name TEXT,
                    Ini_File_Path TEXT,
                    StructureName TEXT,
                    InputFolder TEXT,
                    OutputFolder TEXT,
                    NumReplicates INTEGER,
                    FixedParams TEXT,
                    PriorSummary TEXT,
                    Ini_Hash TEXT,
                    XML_Hash TEXT,
                    Rules_Hash TEXT,
                    Structure_Config_Hash TEXT,
                    Effective_Run_Hash TEXT)""")
    for idx, spec in enumerate(abc_context.models):
        fp = _model_config_fingerprint(spec)
        cursor.execute(
            """INSERT OR REPLACE INTO Models
               (ModelIndex, Name, Ini_File_Path, StructureName, InputFolder, OutputFolder,
                NumReplicates, FixedParams, PriorSummary,
                Ini_Hash, XML_Hash, Rules_Hash, Structure_Config_Hash, Effective_Run_Hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                idx,
                spec.name,
                spec.model_config.get("ini_path"),
                spec.model_config.get("struc_name"),
                spec.model_config.get("input_folder"),
                spec.model_config.get("output_folder"),
                int(spec.num_replicates) if spec.num_replicates is not None else None,
                json.dumps(spec.fixed_params, default=str),
                json.dumps(_prior_summary(spec.prior), default=str),
                fp.get("ini_file_hash"),
                fp.get("xml_file_hash"),
                fp.get("rules_file_hash"),
                fp.get("structure_config_hash"),
                fp.get("effective_run_hash"),
            ),
        )
    conn.commit()
    conn.close()


def insert_model_probabilities_db(db_file: str, history):
    """Store per-population ABC-SMC model probabilities in a ``ModelProbabilities`` table.

    Values come straight from ``pyabc.History.get_model_probabilities()``.
    """
    import sqlite3
    try:
        df = history.get_model_probabilities()
    except Exception:
        return
    if df is None or len(df) == 0:
        return
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS ModelProbabilities (
                    Population INTEGER,
                    ModelIndex INTEGER,
                    Probability DOUBLE,
                    PRIMARY KEY (Population, ModelIndex))""")
    for population, row in df.iterrows():
        for model_index, probability in row.items():
            # column labels may be scalars or 1-tuples ('m',) depending on pyABC version
            m = model_index[-1] if isinstance(model_index, tuple) else model_index
            cursor.execute(
                "INSERT OR REPLACE INTO ModelProbabilities (Population, ModelIndex, Probability) VALUES (?, ?, ?)",
                (int(population), int(m), float(probability)),
            )
    conn.commit()
    conn.close()


def insert_metadata_db(db_file: str, abc_context: "CalibrationContext"):
    """Write the run-level ``Metadata`` row (one row, ``Method='ABC'``).

    ``Ini_File_Path`` / ``StructureName`` describe a single calibrated model. For a
    model-selection run (more than one candidate) they are written as ``NULL`` and
    the per-model configuration lives in the ``Models`` table instead.
    """
    import sqlite3
    model_selection = len(abc_context.models) > 1
    ini_path = None if model_selection else abc_context.models[0].model_config.get('ini_path')
    struc_name = None if model_selection else abc_context.models[0].model_config.get('struc_name')

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    # Create Metadata Table
    cursor.execute(f"""CREATE TABLE IF NOT EXISTS Metadata (
                    Method TEXT PRIMARY KEY,
                    ObsData_Path TEXT,
                    Ini_File_Path TEXT,
                    StructureName TEXT,
                    uq_physicell_version TEXT,
                    pcdl_version TEXT,
                    pyabc_version TEXT)""")
    conn.commit()
    conn.close()
    # Insert the Data
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(f"""INSERT OR REPLACE INTO Metadata (Method, ObsData_Path, Ini_File_Path, StructureName, uq_physicell_version, pcdl_version, pyabc_version) VALUES (?, ?, ?, ?, ?, ?, ?)""", (
        "ABC",
        abc_context.obsData_path,
        ini_path,
        struc_name,
        uq_physicell_version,
        pcdl_version,
        pyabc_version
    ))
    conn.commit()
    conn.close()