
from abc_context.CalibrationContext import CalibrationContext
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

def insert_metadata_db(db_file:str, abc_context:CalibrationContext):
    import sqlite3
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
        abc_context.model_config['ini_path'],
        abc_context.model_config['struc_name'],
        uq_physicell_version,
        pcdl_version,
        pyabc_version
    ))
    conn.commit()
    conn.close()