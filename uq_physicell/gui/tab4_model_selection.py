import numpy as np
import os
import json

# All the specific classes we need
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QListWidget, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# My local modules
from uq_physicell.database.bo_db import load_parameter_space, load_metadata, load_qois, load_gp_models


def add_db_files_to_table(main_window):
    files, _ = QFileDialog.getOpenFileNames(main_window, "Select Calibration .db Files", "", "Database Files (*.db)")
    if not files:
        return
    for file in files:
        row_position = main_window.db_table.rowCount()
        model_id = row_position + 1
        main_window.dic_table[model_id] = {"Database_path": file}
        try:
            main_window.dic_table[model_id]["Num_Params"] = load_parameter_space(file).shape[0]
            main_window.dic_table[model_id]["Max hypervolume"] = np.max(load_gp_models(file)['Hypervolume'])
        except Exception as e:
            main_window.dic_table.pop(model_id, None)
            QMessageBox.warning(main_window, "DB File not supported", f"Error occurred while loading parameters and hypervolumes from {os.path.basename(file)}: {e}")
            return
        if model_id > 1:
            # Check consistency with the previous file
            file_prev = main_window.dic_table[model_id - 1]["Database_path"]
            try:
                check_consistency_dbs(load_metadata(file_prev), load_metadata(file), load_qois(file_prev), load_qois(file))
            except ValueError as e:
                main_window.dic_table.pop(model_id, None)
                QMessageBox.warning(main_window, "Inconsistent DB Files", f"Database file {os.path.basename(file)} has inconsistent QoIs or BO setup compared to previous files: {e}")
                return 

        main_window.db_table.insertRow(row_position)
        # Model number (auto-increment)
        main_window.db_table.setItem(row_position, 0, QTableWidgetItem(str(model_id)))
        # Database name
        main_window.db_table.setItem(row_position, 1, QTableWidgetItem(os.path.basename(file)))
        # Number of Parameters
        main_window.db_table.setItem(row_position, 2, QTableWidgetItem(str(main_window.dic_table[model_id]["Num_Params"])))
        # Hypervolume
        main_window.db_table.setItem(row_position, 3, QTableWidgetItem(f"{main_window.dic_table[model_id]['Max hypervolume']:.4e}"))
        # Add remove button
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(lambda _, r=row_position: remove_db_row(main_window, r))
        main_window.db_table.setCellWidget(row_position, 4, remove_btn)

def remove_db_row(main_window, row):
    main_window.db_table.removeRow(row)
    main_window.dic_table.pop(row + 1, None)  # Remove from dictionary (row + 1 because model_id starts from 1)
    # Re-number Model column
    for i in range(main_window.db_table.rowCount()):
        main_window.db_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))

def check_consistency_dbs(df_metadata_prev, df_metadata, df_qois_prev, df_qois):
    """
    Check if the parameter space and BO setup are consistent between two database files.
    Raises a ValueError if they are not consistent."""
    if 'BO_Options' not in df_metadata.columns or 'BO_Options' not in df_metadata_prev.columns:
        raise ValueError("Missing BO setup (BO_Options) in the database file.")
    bo_setup_current = df_metadata['BO_Options'].iloc[0]
    bo_setup_prev = df_metadata_prev['BO_Options'].iloc[0]
    if bo_setup_current is None or bo_setup_prev is None:
        raise ValueError("Missing BO setup (BO_Options) in the database file.")
    # Extract the BO setup dictionary from the metadata for both current and previous files
    bo_setup_current = json.loads(bo_setup_current)
    bo_setup_prev = json.loads(bo_setup_prev)

    # Check the parameter space and BO options for consistency
    if bo_setup_prev != bo_setup_current:
        raise ValueError(f"Inconsistent BO setup between files. Previous: {bo_setup_prev}, Current: {bo_setup_current}")
    if not df_qois.equals(df_qois_prev):
        raise ValueError(f"Inconsistent QoIs between files. Previous: {df_qois_prev}, Current: {df_qois}")
    
def plot_complexity_vs_hypervolume(main_window):
    if main_window.db_table.rowCount() == 0:
        QMessageBox.warning(main_window, "No DB Files", "Please load at least one calibration .db file.")
        return
    
    # Plot Comparison
    main_window.comparison_figure.clear()
    ax = main_window.comparison_figure.add_subplot(111)
    max_param = 1; min_param = 0
    for model_id, dict_info in main_window.dic_table.items():
        ax.scatter(dict_info["Num_Params"], dict_info["Max hypervolume"], label=f"Model {model_id}")
        if dict_info["Num_Params"] > max_param:
            max_param = dict_info["Num_Params"]
        if dict_info["Num_Params"] < min_param:
            min_param = dict_info["Num_Params"]
    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Hypervolume')
    # Integer ticks from min_param to max_param, so a single loaded model still shows a meaningful scale
    ax.set_xticks(range(min_param, max_param + 2, 1))
    ax.set_xlim(min_param - 0.5, max_param + 0.5)
    ax.legend()
    main_window.comparison_canvas.draw()


def create_tab4(main_window):
    layout_tab4 = QVBoxLayout()
    # Title
    load_dbs_label = QLabel("<b>Load Multiple Bayesian Optimization (BO) Calibration .db Files</b>")
    load_dbs_label.setAlignment(Qt.AlignCenter)
    layout_tab4.addWidget(load_dbs_label)

    # Load button
    load_db_btn = QPushButton("Add Calibration .db Files")
    load_db_btn.setStyleSheet("background-color: lightgreen; color: black")
    load_db_btn.clicked.connect(lambda: add_db_files_to_table(main_window))
    layout_tab4.addWidget(load_db_btn)

    # Process and plot button
    process_btn = QPushButton("Plot Complexity vs. Hypervolume")
    process_btn.setStyleSheet("background-color: lightgreen; color: black")
    process_btn.clicked.connect(lambda: plot_complexity_vs_hypervolume(main_window))
    layout_tab4.addWidget(process_btn)

    # Table for models
    main_window.dic_table = {}
    main_window.db_table = QTableWidget()
    main_window.db_table.setColumnCount(5)
    main_window.db_table.setHorizontalHeaderLabels(["Model", "Database", "Number of Parameters", "Max hypervolume", "Remove"])
    header = main_window.db_table.horizontalHeader()
    for col in range(main_window.db_table.columnCount()):
        header.setSectionResizeMode(col, QHeaderView.Stretch)
    layout_tab4.addWidget(main_window.db_table)
     # Set gray background for header
    header = main_window.db_table.horizontalHeader()
    header.setStyleSheet("QHeaderView::section { background-color: lightgray; color: black; font-weight: bold; }")

    # Matplotlib Figure for Comparison plot
    plot_label = QLabel("<b>Model Comparison - Complexity vs. Hypervolume</b>")
    plot_label.setAlignment(Qt.AlignCenter)
    layout_tab4.addWidget(plot_label)
    main_window.comparison_figure = Figure(figsize=(5,3), layout="tight")
    main_window.comparison_canvas = FigureCanvas(main_window.comparison_figure)
    layout_tab4.addWidget(main_window.comparison_canvas)
    return layout_tab4