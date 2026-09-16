"""
Test script for ABC CalibrationContext to verify basic functionality.
This tests the import and initialization without requiring a full PhysiCell setup.
"""

import sys
import os
import logging
import tempfile
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

# Add the path to import uq_physicell
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that the new CalibrationContext can be imported."""
    from uq_physicell.abc import CalibrationContext
    from pyabc import Distribution, RV
    # Basic sanity assertions: imports resolved
    assert CalibrationContext is not None
    assert Distribution is not None
    assert RV is not None

def test_initialization():
    """Test CalibrationContext initialization with minimal configuration."""
    from uq_physicell.abc import CalibrationContext
    from pyabc import Distribution, RV

    # Setup minimal configuration
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Synthetic observed data
    obs_data = {
        'QoI1': np.array([1.0, 1.1, 1.2, 1.3, 1.4]),
        'QoI2': np.array([0.2, 0.21, 0.22, 0.23, 0.24])
    }

    obs_data_columns = {
        'QoI1': 'QoI1_data',
        'QoI2': 'QoI2_data'
    }

    # Create a temporary INI file for testing
    ini_content = """[model_struc_name]
    numReplicates = 1
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as ini_file:
        ini_file.write(ini_content)
        ini_file_path = ini_file.name

    model_config = {
        'ini_path': ini_file_path,
        'struc_name': 'model_struc_name',
    }

    qoi_functions = {
        'QoI1': 'lambda df: df["QoI1"].values',
        'QoI2': 'lambda df: df["QoI2"].values'
    }

    distance_functions = {
        'QoI1': {'function': 'euclidean', 'weight': 1.0},
        'QoI2': {'function': 'euclidean', 'weight': 1.0}
    }

    lb1 = 0.0; ub1 = 5.0
    lb2 = 0.0; ub2 = 10.0; loc2 = 5.0; scale2 = 2.0 # lb and ub are bounds, loc is mean, scale is stddev
    prior = Distribution(
        param1 = RV('uniform', lb1, ub1-lb1),
        param2 = RV('truncnorm', lb2, ub2, loc2, scale2)
    )

    abc_options = {
        'max_populations': 5,
        'max_simulations': 50,
        'sampler': 'multicore',
        'num_workers': 2,
        'mode': 'local'
    }

    # Create temporary database file
    db_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name

        # Initialize CalibrationContext
        calib_context = CalibrationContext(
            db_path=db_path,
            obsData=obs_data,
            obsData_columns=obs_data_columns,
            model_config=model_config,
            qoi_functions=qoi_functions,
            distance_functions=distance_functions,
            prior=prior,
            abc_options=abc_options,
            logger=logger
        )

        # Sanity-check a few attributes
        assert hasattr(calib_context, 'db_path')
        assert list(calib_context.qoi_functions.keys()) == ['QoI1', 'QoI2']
        # prior is a pyabc Distribution — ensure parameter names property exists if available
        if hasattr(calib_context.prior, 'get_parameter_names'):
            assert callable(calib_context.prior.get_parameter_names)
        assert hasattr(calib_context, 'sampler_type')
        assert hasattr(calib_context, 'num_workers')

    finally:
        # Cleanup
        if db_path and os.path.exists(db_path):
            try:
                os.unlink(db_path)
            except Exception:
                pass
        if ini_file_path and os.path.exists(ini_file_path):
            try:
                os.unlink(ini_file_path)
            except Exception:
                pass

def test_configuration_validation():
    """Test that configuration validation works correctly."""
    from uq_physicell.abc import CalibrationContext
    from pyabc import Distribution, RV

    # Test with missing required keys
    invalid_model_config = {
        'numReplicates': 2
        # Missing 'config_file' and 'model_name'
    }

    with pytest.raises(ValueError):
        CalibrationContext(
            db_path="dummy.db",
            obsData={'dummy': [1, 2, 3]},
            obsData_columns={'dummy': 'dummy'},
            model_config=invalid_model_config,  # Invalid config
            qoi_functions={'dummy': 'lambda x: x'},
            distance_functions={'dummy': {'function': 'euclidean', 'weight': 1.0}},
            prior= Distribution(param1 = RV('uniform', 0, 1.0)),
            abc_options={'max_populations': 5}
        )

def test_model_selection_specs():
    """abc_options['models'] builds one ModelSpec per candidate with per-model IO subfolders."""
    from uq_physicell.abc import CalibrationContext, ModelSpec
    from pyabc import Distribution, RV

    ini_content = "[strucA]\nnumReplicates = 4\n\n[strucB]\nnumReplicates = 2\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as ini_file:
        ini_file.write(ini_content)
        ini_path = ini_file.name

    obs_data = {'QoI1': np.array([1.0, 1.1, 1.2])}
    obs_data_columns = {'QoI1': 'QoI1_data'}
    qoi_functions = {'QoI1': 'lambda df: df["QoI1"].values'}
    distance_functions = {'QoI1': {'function': 'euclidean', 'weight': 1.0}}
    prior_a = Distribution(param1=RV('uniform', 0, 1.0))
    prior_b = Distribution(param1=RV('uniform', 0, 1.0), param2=RV('uniform', 0, 2.0))

    try:
        ctx = CalibrationContext(
            db_path="dummy.db",
            obsData=obs_data,
            obsData_columns=obs_data_columns,
            qoi_functions=qoi_functions,
            distance_functions=distance_functions,
            abc_options={
                'max_populations': 2,
                'sampler': 'multicore',
                'num_workers': 2,
                'models': [
                    {'name': 'hypA', 'model_config': {'ini_path': ini_path, 'struc_name': 'strucA'},
                     'prior': prior_a},
                    {'name': 'hypB', 'model_config': {'ini_path': ini_path, 'struc_name': 'strucB'},
                     'prior': prior_b, 'fixed_params': {'k': 1.0}, 'num_replicates': 3},
                ],
            },
        )

        assert ctx.model_selection is True
        assert ctx.num_models == 2
        assert [m.name for m in ctx.models] == ['hypA', 'hypB']
        assert isinstance(ctx.models[0], ModelSpec)
        # replicate counts: from ini / explicit override
        assert ctx.models[0].num_replicates == 4
        assert ctx.models[1].num_replicates == 3
        # per-model IO isolation auto-applied for multi-model runs
        assert ctx.models[0].model_config['output_folder'] == 'hypA/'
        assert ctx.models[1].model_config['input_folder'] == 'hypB/'
        # back-compat accessors delegate to models[0]
        assert ctx.prior is ctx.models[0].prior
        assert ctx.fixed_params == {}
        # one pyABC wrapper per model, uniquely named
        names = {ctx.create_model_wrapper(m, ctx.workers_inner).__name__ for m in ctx.models}
        assert names == {'run_physicell_hypA', 'run_physicell_hypB'}
    finally:
        if os.path.exists(ini_path):
            os.unlink(ini_path)


def test_model_selection_validation():
    """Invalid model-selection specs raise ValueError."""
    from uq_physicell.abc import CalibrationContext
    from pyabc import Distribution, RV

    common = dict(
        db_path="dummy.db",
        obsData={'QoI1': [1, 2, 3]},
        obsData_columns={'QoI1': 'QoI1'},
        qoi_functions={'QoI1': 'lambda x: x'},
        distance_functions={'QoI1': {'function': 'euclidean', 'weight': 1.0}},
    )
    prior = Distribution(param1=RV('uniform', 0, 1.0))
    mc = {'ini_path': 'x.ini', 'struc_name': 's', 'numReplicates': 1}

    # neither models nor model_config/prior
    with pytest.raises(ValueError):
        CalibrationContext(abc_options={}, **common)
    # model entry missing required 'prior'
    with pytest.raises(ValueError):
        CalibrationContext(abc_options={'models': [{'name': 'a', 'model_config': mc}]}, **common)
    # non-unique names
    with pytest.raises(ValueError):
        CalibrationContext(abc_options={'models': [
            {'name': 'a', 'model_config': mc, 'prior': prior},
            {'name': 'a', 'model_config': mc, 'prior': prior},
        ]}, **common)
    # unsafe name
    with pytest.raises(ValueError):
        CalibrationContext(abc_options={'models': [
            {'name': 'a/b', 'model_config': mc, 'prior': prior},
        ]}, **common)


def test_model_selection_db_tables():
    """Metadata NULLs the single-model columns for a selection run; Models has one row per candidate."""
    import sqlite3
    from uq_physicell.abc import CalibrationContext
    from uq_physicell.abc.utils import insert_metadata_db, insert_models_db
    from pyabc import Distribution, RV

    ini_content = "[strucA]\nnumReplicates = 2\n\n[strucB]\nnumReplicates = 2\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as ini_file:
        ini_file.write(ini_content)
        ini_path = ini_file.name
    db_path = tempfile.NamedTemporaryFile(suffix='.db', delete=False).name

    common = dict(
        obsData={'QoI1': np.array([1.0, 1.1])},
        obsData_columns={'QoI1': 'QoI1_data'},
        qoi_functions={'QoI1': 'lambda df: df["QoI1"].values'},
        distance_functions={'QoI1': {'function': 'euclidean', 'weight': 1.0}},
    )
    prior = Distribution(param1=RV('uniform', 0, 1.0))

    try:
        # single model -> Metadata carries ini/struc
        ctx1 = CalibrationContext(db_path=db_path, model_config={'ini_path': ini_path, 'struc_name': 'strucA'},
                                  prior=prior, abc_options={}, **common)
        insert_metadata_db(db_path, ctx1)
        insert_models_db(db_path, ctx1)
        conn = sqlite3.connect(db_path)
        assert conn.execute("SELECT Ini_File_Path, StructureName FROM Metadata").fetchone() == (ini_path, 'strucA')
        assert conn.execute("SELECT COUNT(*) FROM Models").fetchone()[0] == 1
        conn.close()

        # model selection -> Metadata NULLs ini/struc, Models has both rows
        ctx2 = CalibrationContext(db_path=db_path, abc_options={'models': [
            {'name': 'A', 'model_config': {'ini_path': ini_path, 'struc_name': 'strucA'}, 'prior': prior},
            {'name': 'B', 'model_config': {'ini_path': ini_path, 'struc_name': 'strucB'}, 'prior': prior},
        ]}, **common)
        insert_metadata_db(db_path, ctx2)
        insert_models_db(db_path, ctx2)
        conn = sqlite3.connect(db_path)
        assert conn.execute("SELECT Ini_File_Path, StructureName FROM Metadata").fetchone() == (None, None)
        rows = conn.execute("SELECT ModelIndex, Name, StructureName FROM Models ORDER BY ModelIndex").fetchall()
        assert rows == [(0, 'A', 'strucA'), (1, 'B', 'strucB')]
        # fingerprint columns exist (values may be NULL when the model can't be instantiated)
        cols = {c[1] for c in conn.execute("PRAGMA table_info(Models)")}
        assert {'Ini_Hash', 'XML_Hash', 'Rules_Hash', 'Structure_Config_Hash', 'Effective_Run_Hash'} <= cols
        conn.close()
    finally:
        for p in (ini_path, db_path):
            if os.path.exists(p):
                os.unlink(p)


def test_sequential_replicate_aggregation_and_seeds():
    """Regression test for two bugs in _run_physicell_model_sequential:

    1. The replicate loop reused the `replicate_id` parameter as its loop
       variable, shadowing it -- so `if replicate_id is None:` was never true
       after the loop ran, and aggregation_func was silently skipped whenever
       this is called with replicate_id=None (i.e. the non-multicore / "dask"
       sampler path).
    2. Replicates were run without a distinct random_seed per replicate, so
       parallel replicates of the same particle could collide on PhysiCell's
       default system-clock seed.
    """
    from uq_physicell.abc import CalibrationContext
    from uq_physicell.abc.abc_context import ModelSpec
    from pyabc import Distribution, RV

    ini_content = "[strucA]\nnumReplicates = 3\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as ini_file:
        ini_file.write(ini_content)
        ini_path = ini_file.name
    db_path = tempfile.NamedTemporaryFile(suffix='.db', delete=False).name
    prior = Distribution(param1=RV('uniform', 0, 1.0))

    try:
        calib_context = CalibrationContext(
            db_path=db_path,
            obsData={'QoI1': np.array([1.0, 1.1])},
            obsData_columns={'QoI1': 'QoI1_data'},
            qoi_functions={'QoI1': 'lambda df: df["QoI1"].values'},
            distance_functions={'QoI1': {'function': 'euclidean', 'weight': 1.0}},
            model_config={'ini_path': ini_path, 'struc_name': 'strucA'},
            prior=prior,
            abc_options={},
        )
        model_spec = ModelSpec(
            name='strucA',
            model_config={'ini_path': ini_path, 'struc_name': 'strucA'},
            prior=prior,
            num_replicates=3,
        )

        fake_model = MagicMock()
        fake_model.XML_parameters_variable = {}
        fake_model.parameters_rules_variable = {}

        aggregation_calls = []
        def fake_aggregation_func(dic_all_replicates):
            aggregation_calls.append(dic_all_replicates)
            return "AGGREGATED"

        seen_seeds = []
        def fake_run_replicate_serializable(**kwargs):
            seen_seeds.append(kwargs["random_seed"])
            return (None, None, pd.DataFrame({"sampleID": [1], "time": [0]}))

        with patch.object(calib_context, "_instantiate_model", return_value=fake_model), \
             patch.object(calib_context, "aggregation_func", side_effect=fake_aggregation_func), \
             patch("uq_physicell.abc.abc_context.run_replicate_serializable", side_effect=fake_run_replicate_serializable):
            result = calib_context._run_physicell_model_sequential(
                pars={}, model_spec=model_spec, sample_id=1, replicate_id=None
            )

        assert aggregation_calls, "aggregation_func was never called -- replicate_id shadowing regression"
        assert result == "AGGREGATED"

        assert len(seen_seeds) == 3
        assert len(set(seen_seeds)) == 3, "replicates must not share a random seed"
    finally:
        for p in (ini_path, db_path):
            if os.path.exists(p):
                os.unlink(p)


def main():
    """Run all tests."""
    print("🧪 Testing ABC CalibrationContext")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Initialization Test", test_initialization),
        ("Validation Test", test_configuration_validation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        result = test_func()
        results.append(result)
        print(f"{'✅ PASSED' if result else '❌ FAILED'}: {test_name}")
    
    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! CalibrationContext is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)