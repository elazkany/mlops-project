```
mlops-project
|-- Dockerfile
|-- LICENSE
|-- Makefile
|-- README.md
|-- __pycache__
|   `-- test_path.cpython-310.pyc
|-- app
|   |-- __init__.py
|   |-- __pycache__
|   |   |-- __init__.cpython-310.pyc
|   |   `-- app.cpython-310.pyc
|   `-- app.py
|-- bin
|   `-- setup.sh
|-- data
|   |-- last_updated.txt
|   |-- processed
|   |   `-- processed_creditcard.csv
|   |-- raw
|   |   `-- creditcard.csv
|   |-- test
|   |   |-- X_test.json
|   |   `-- y_test.json
|   |-- train
|   |   |-- train_original.npz
|   |   `-- train_resampled.npz
|   `-- validation
|       `-- valid.npz
|-- deployment
|   |-- model_artifacts
|   |   |-- MLmodel
|   |   |-- conda.yaml
|   |   |-- input_example.json
|   |   |-- metadata.json
|   |   |-- model.pkl
|   |   |-- python_env.yaml
|   |   |-- requirements.txt
|   |   `-- serving_input_example.json
|   `-- model_version.json
|-- dvc.lock
|-- dvc.yaml
|-- htmlcov
|   |-- class_index.html
|   |-- coverage_html.js
|   |-- coverage_html_cb_6fb7b396.js
|   |-- d_0805fb2174612f99___init___py.html
|   |-- d_0805fb2174612f99_export_model_py.html
|   |-- d_145eef247bfb46b6___init___py.html
|   |-- d_6cf9dd5166a37b1e___init___py.html
|   |-- d_6cf9dd5166a37b1e_challenger_model_py.html
|   |-- d_6cf9dd5166a37b1e_generate_cml_report_py.html
|   |-- d_6cf9dd5166a37b1e_run_train_py.html
|   |-- d_6cf9dd5166a37b1e_split_py.html
|   |-- d_6cf9dd5166a37b1e_train_and_evaluate_py.html
|   |-- d_6cf9dd5166a37b1e_train_py.html
|   |-- d_defad1a87623d8bf___init___py.html
|   |-- d_defad1a87623d8bf_model_py.html
|   |-- d_e5ccf62099f28249___init___py.html
|   |-- d_e5ccf62099f28249_check_for_new_data_py.html
|   |-- d_e5ccf62099f28249_collect_and_extract_py.html
|   |-- d_e5ccf62099f28249_preprocess_py.html
|   |-- d_f1b38b22aeb65474___init___py.html
|   |-- d_f1b38b22aeb65474_io_load_py.html
|   |-- d_f1b38b22aeb65474_io_save_py.html
|   |-- d_f1b38b22aeb65474_params_py.html
|   |-- d_f1b38b22aeb65474_plot_utils_py.html
|   |-- favicon_32.png
|   |-- favicon_32_cb_58284776.png
|   |-- function_index.html
|   |-- index.html
|   |-- keybd_closed.png
|   |-- keybd_closed_cb_ce680311.png
|   |-- keybd_open.png
|   |-- status.json
|   |-- style.css
|   |-- style_cb_81f8c14c.css
|   |-- z_145eef247bfb46b6___init___py.html
|   |-- z_e5ccf62099f28249___init___py.html
|   |-- z_e5ccf62099f28249_check_for_new_data_py.html
|   |-- z_e5ccf62099f28249_check_processed_py.html
|   |-- z_e5ccf62099f28249_collect_and_extract_py.html
|   `-- z_e5ccf62099f28249_preprocess_py.html
|-- mlruns
|   |-- 0
|   |   `-- meta.yaml
|   |-- 257253410539376350
|   |   |-- 11d23b2c707241218b0ce7a0a59011f4
|   |   |   |-- artifacts
|   |   |   |   `-- confusion_matrix.png
|   |   |   |-- meta.yaml
|   |   |   |-- metrics
|   |   |   |   |-- accuracy
|   |   |   |   |-- f1_score
|   |   |   |   |-- precision
|   |   |   |   `-- recall
|   |   |   |-- outputs
|   |   |   |   `-- m-cb904c08e3284cb586ad66339b6baf01
|   |   |   |       `-- meta.yaml
|   |   |   |-- params
|   |   |   |   |-- max_depth
|   |   |   |   |-- n_estimators
|   |   |   |   `-- random_state
|   |   |   `-- tags
|   |   |       |-- mlflow.runName
|   |   |       |-- mlflow.source.git.commit
|   |   |       |-- mlflow.source.name
|   |   |       |-- mlflow.source.type
|   |   |       |-- mlflow.user
|   |   |       |-- model_type
|   |   |       `-- run_name
|   |   |-- 85ac4d8bbdc944ec962aed16163144d4
|   |   |   |-- artifacts
|   |   |   |   `-- confusion_matrix.png
|   |   |   |-- meta.yaml
|   |   |   |-- metrics
|   |   |   |   |-- accuracy
|   |   |   |   |-- f1_score
|   |   |   |   |-- precision
|   |   |   |   `-- recall
|   |   |   |-- outputs
|   |   |   |   `-- m-45a28cfd795c459b9149da7cfb4f3d71
|   |   |   |       `-- meta.yaml
|   |   |   |-- params
|   |   |   |   |-- max_depth
|   |   |   |   |-- n_estimators
|   |   |   |   `-- random_state
|   |   |   `-- tags
|   |   |       |-- mlflow.runName
|   |   |       |-- mlflow.source.git.commit
|   |   |       |-- mlflow.source.name
|   |   |       |-- mlflow.source.type
|   |   |       |-- mlflow.user
|   |   |       |-- model_type
|   |   |       `-- run_name
|   |   |-- 9ea15172457a4fbe9c09dc5312f8b00d
|   |   |   |-- artifacts
|   |   |   |   `-- confusion_matrix.png
|   |   |   |-- meta.yaml
|   |   |   |-- metrics
|   |   |   |   |-- accuracy
|   |   |   |   |-- f1_score
|   |   |   |   |-- precision
|   |   |   |   `-- recall
|   |   |   |-- outputs
|   |   |   |   `-- m-5ffc1037be334a9ebd7e20430c687b63
|   |   |   |       `-- meta.yaml
|   |   |   |-- params
|   |   |   |   |-- C
|   |   |   |   `-- penalty
|   |   |   `-- tags
|   |   |       |-- mlflow.runName
|   |   |       |-- mlflow.source.git.commit
|   |   |       |-- mlflow.source.name
|   |   |       |-- mlflow.source.type
|   |   |       |-- mlflow.user
|   |   |       |-- model_type
|   |   |       `-- run_name
|   |   |-- a82a588b945d43d79d175fd65b4db16d
|   |   |   |-- artifacts
|   |   |   |   `-- confusion_matrix.png
|   |   |   |-- meta.yaml
|   |   |   |-- metrics
|   |   |   |   |-- accuracy
|   |   |   |   |-- f1_score
|   |   |   |   |-- precision
|   |   |   |   `-- recall
|   |   |   |-- outputs
|   |   |   |   `-- m-61c43f2519a743f2ab3e19f0d810f467
|   |   |   |       `-- meta.yaml
|   |   |   |-- params
|   |   |   |   |-- C
|   |   |   |   `-- penalty
|   |   |   `-- tags
|   |   |       |-- mlflow.runName
|   |   |       |-- mlflow.source.git.commit
|   |   |       |-- mlflow.source.name
|   |   |       |-- mlflow.source.type
|   |   |       |-- mlflow.user
|   |   |       |-- model_type
|   |   |       `-- run_name
|   |   |-- meta.yaml
|   |   `-- models
|   |       |-- m-45a28cfd795c459b9149da7cfb4f3d71
|   |       |   |-- artifacts
|   |       |   |   |-- MLmodel
|   |       |   |   |-- conda.yaml
|   |       |   |   |-- input_example.json
|   |       |   |   |-- model.pkl
|   |       |   |   |-- python_env.yaml
|   |       |   |   |-- requirements.txt
|   |       |   |   `-- serving_input_example.json
|   |       |   |-- meta.yaml
|   |       |   |-- metrics
|   |       |   |   |-- accuracy
|   |       |   |   |-- f1_score
|   |       |   |   |-- precision
|   |       |   |   `-- recall
|   |       |   |-- params
|   |       |   |   |-- max_depth
|   |       |   |   |-- n_estimators
|   |       |   |   `-- random_state
|   |       |   `-- tags
|   |       |       |-- mlflow.source.git.commit
|   |       |       |-- mlflow.source.name
|   |       |       |-- mlflow.source.type
|   |       |       `-- mlflow.user
|   |       |-- m-5ffc1037be334a9ebd7e20430c687b63
|   |       |   |-- artifacts
|   |       |   |   |-- MLmodel
|   |       |   |   |-- conda.yaml
|   |       |   |   |-- input_example.json
|   |       |   |   |-- model.pkl
|   |       |   |   |-- python_env.yaml
|   |       |   |   |-- requirements.txt
|   |       |   |   `-- serving_input_example.json
|   |       |   |-- meta.yaml
|   |       |   |-- metrics
|   |       |   |   |-- accuracy
|   |       |   |   |-- f1_score
|   |       |   |   |-- precision
|   |       |   |   `-- recall
|   |       |   |-- params
|   |       |   |   |-- C
|   |       |   |   `-- penalty
|   |       |   `-- tags
|   |       |       |-- mlflow.source.git.commit
|   |       |       |-- mlflow.source.name
|   |       |       |-- mlflow.source.type
|   |       |       `-- mlflow.user
|   |       |-- m-61c43f2519a743f2ab3e19f0d810f467
|   |       |   |-- artifacts
|   |       |   |   |-- MLmodel
|   |       |   |   |-- conda.yaml
|   |       |   |   |-- input_example.json
|   |       |   |   |-- model.pkl
|   |       |   |   |-- python_env.yaml
|   |       |   |   |-- requirements.txt
|   |       |   |   `-- serving_input_example.json
|   |       |   |-- meta.yaml
|   |       |   |-- metrics
|   |       |   |   |-- accuracy
|   |       |   |   |-- f1_score
|   |       |   |   |-- precision
|   |       |   |   `-- recall
|   |       |   |-- params
|   |       |   |   |-- C
|   |       |   |   `-- penalty
|   |       |   `-- tags
|   |       |       |-- mlflow.source.git.commit
|   |       |       |-- mlflow.source.name
|   |       |       |-- mlflow.source.type
|   |       |       `-- mlflow.user
|   |       `-- m-cb904c08e3284cb586ad66339b6baf01
|   |           |-- artifacts
|   |           |   |-- MLmodel
|   |           |   |-- conda.yaml
|   |           |   |-- input_example.json
|   |           |   |-- model.pkl
|   |           |   |-- python_env.yaml
|   |           |   |-- requirements.txt
|   |           |   `-- serving_input_example.json
|   |           |-- meta.yaml
|   |           |-- metrics
|   |           |   |-- accuracy
|   |           |   |-- f1_score
|   |           |   |-- precision
|   |           |   `-- recall
|   |           |-- params
|   |           |   |-- max_depth
|   |           |   |-- n_estimators
|   |           |   `-- random_state
|   |           `-- tags
|   |               |-- mlflow.modelVersions
|   |               |-- mlflow.source.git.commit
|   |               |-- mlflow.source.name
|   |               |-- mlflow.source.type
|   |               `-- mlflow.user
|   `-- models
|       `-- challenger_model
|           |-- aliases
|           |   `-- prod
|           |-- meta.yaml
|           `-- version-1
|               `-- meta.yaml
|-- notebooks
|   `-- ml_baseline.ipynb
|-- params.yaml
|-- plots
|   |-- predictions.csv
|   `-- roc_curve.csv
|-- reports
|   |-- best_model_confusion_matrix.png
|   |-- best_model_metrics.json
|   |-- compare_confusion_matrices.png
|   `-- compare_metrics.json
|-- requirements.txt
|-- setup.cfg
|-- src
|   |-- __init__.py
|   |-- __pycache__
|   |   `-- __init__.cpython-310.pyc
|   |-- data
|   |   |-- __init__.py
|   |   |-- __pycache__
|   |   |   |-- __init__.cpython-310.pyc
|   |   |   |-- check_for_new_data.cpython-310.pyc
|   |   |   |-- collect_and_extract.cpython-310.pyc
|   |   |   `-- preprocess.cpython-310.pyc
|   |   |-- check_for_new_data.py
|   |   |-- collect_and_extract.py
|   |   `-- preprocess.py
|   |-- deploy
|   |   |-- __init__.py
|   |   |-- __pycache__
|   |   |   |-- __init__.cpython-310.pyc
|   |   |   `-- export_model.cpython-310.pyc
|   |   `-- export_model.py
|   |-- models
|   |   |-- __init__.py
|   |   |-- __pycache__
|   |   |   |-- __init__.cpython-310.pyc
|   |   |   `-- model.cpython-310.pyc
|   |   `-- model.py
|   |-- train
|   |   |-- __init__.py
|   |   |-- __pycache__
|   |   |   |-- __init__.cpython-310.pyc
|   |   |   |-- challenger_model.cpython-310.pyc
|   |   |   |-- generate_cml_report.cpython-310.pyc
|   |   |   |-- run_train.cpython-310.pyc
|   |   |   |-- split.cpython-310.pyc
|   |   |   |-- train.cpython-310.pyc
|   |   |   `-- train_and_evaluate.cpython-310.pyc
|   |   |-- challenger_model.py
|   |   |-- generate_cml_report.py
|   |   |-- run_train.py
|   |   |-- split.py
|   |   `-- train_and_evaluate.py
|   `-- utils
|       |-- __init__.py
|       |-- __pycache__
|       |   |-- __init__.cpython-310.pyc
|       |   |-- io_load.cpython-310.pyc
|       |   |-- io_save.cpython-310.pyc
|       |   |-- params.cpython-310.pyc
|       |   `-- plot_utils.cpython-310.pyc
|       |-- io_load.py
|       |-- io_save.py
|       `-- plot_utils.py
`-- tests
    |-- __init__.py
    |-- __pycache__
    |   `-- __init__.cpython-310.pyc
    |-- data
    |   |-- __init__.py
    |   |-- __pycache__
    |   |   |-- __init__.cpython-310.pyc
    |   |   |-- test_check_for_new_data.cpython-310-pytest-8.4.1.pyc
    |   |   |-- test_collect_and_extract.cpython-310-pytest-8.4.1.pyc
    |   |   `-- test_preprocess.cpython-310-pytest-8.4.1.pyc
    |   |-- test_check_for_new_data.py
    |   |-- test_collect_and_extract.py
    |   `-- test_preprocess.py
    |-- deploy
    |   |-- __init__.py
    |   |-- __pycache__
    |   |   |-- __init__.cpython-310.pyc
    |   |   `-- test_export_model.cpython-310-pytest-8.4.1.pyc
    |   `-- test_export_model.py
    |-- models
    |   |-- __init__.py
    |   |-- __pycache__
    |   |   |-- __init__.cpython-310.pyc
    |   |   `-- test_model.cpython-310-pytest-8.4.1.pyc
    |   `-- test_model.py
    |-- train
    |   |-- __pycache__
    |   |   |-- test_challenger_model.cpython-310-pytest-8.4.1.pyc
    |   |   |-- test_generate_cml_report.cpython-310-pytest-8.4.1.pyc
    |   |   |-- test_run_train.cpython-310-pytest-8.4.1.pyc
    |   |   |-- test_split.cpython-310-pytest-8.4.1.pyc
    |   |   `-- test_train_and_evaluate.cpython-310-pytest-8.4.1.pyc
    |   |-- test_challenger_model.py
    |   |-- test_generate_cml_report.py
    |   |-- test_run_train.py
    |   |-- test_split.py
    |   `-- test_train_and_evaluate.py
    `-- utils
        |-- __init__.py
        |-- __pycache__
        |   |-- __init__.cpython-310.pyc
        |   |-- test_io_load.cpython-310-pytest-8.4.1.pyc
        |   |-- test_io_save.cpython-310-pytest-8.4.1.pyc
        |   `-- test_plot_utils.cpython-310-pytest-8.4.1.pyc
        |-- test_io_load.py
        |-- test_io_save.py
        `-- test_plot_utils.py
```