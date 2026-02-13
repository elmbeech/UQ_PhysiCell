
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