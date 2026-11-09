"""Patch the notebook cells to use local jars and SQLite MLflow."""
import json

NB_PATH = r"C:\Users\21625\Desktop\bigdata\dl_advanced_risk_model.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    cid = cell.get("id", "")

    # Fix env cell - add PATH to hadoop\bin
    if cid == "aa000002":
        cell["source"] = [
            "# ===========================================================\n",
            "# SECTION 1 \u2013 Environment (run BEFORE any PySpark import)\n",
            "# ===========================================================\n",
            "import os, pathlib, sys\n",
            "\n",
            'HADOOP_HOME = r"C:\\Users\\21625\\Desktop\\bigdata\\hadoop"\n',
            'SPARK_TMP   = r"C:\\Users\\21625\\Desktop\\bigdata\\spark_temp"\n',
            'JAVA_HOME   = r"C:\\Program Files\\Microsoft\\jdk-17.0.21.21-hotspot"\n',
            "\n",
            'os.environ["HADOOP_HOME"]      = HADOOP_HOME\n',
            'os.environ["JAVA_HOME"]        = JAVA_HOME\n',
            'os.environ["SPARK_LOCAL_DIRS"] = SPARK_TMP\n',
            'os.environ["PATH"] = HADOOP_HOME + r"\\bin;" + os.environ.get("PATH", "")\n',
            "\n",
            'hadoop_fwd = HADOOP_HOME.replace("\\\\", "/")\n',
            'os.environ["PYSPARK_SUBMIT_ARGS"] = (\n',
            """    f'--driver-java-options "-Dhadoop.home.dir={hadoop_fwd}" pyspark-shell'\n""",
            ")\n",
            "\n",
            "from pathlib import Path\n",
            "PROJECT_ROOT = Path.cwd()\n",
            'MODEL_DIR    = PROJECT_ROOT / "models"\n',
            'REPORT_DIR   = PROJECT_ROOT / "data" / "reports"\n',
            "MODEL_DIR.mkdir(parents=True, exist_ok=True)\n",
            "REPORT_DIR.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            'print("PYSPARK_SUBMIT_ARGS:", os.environ["PYSPARK_SUBMIT_ARGS"])\n',
            'print("winutils.exe:", pathlib.Path(HADOOP_HOME, "bin", "winutils.exe").is_file())\n',
            'print("PROJECT_ROOT:", PROJECT_ROOT)\n',
        ]
        print("  Fixed env cell (aa000002)")

    # Fix Spark session cell - use local jars
    if cid == "aa000004":
        cell["source"] = [
            "# ===========================================================\n",
            "# SECTION 3 \u2013 Spark session\n",
            "# ===========================================================\n",
            "from pyspark.sql import SparkSession, functions as F\n",
            "from pyspark.sql.window import Window\n",
            "\n",
            "# Use pre-downloaded Delta Lake jars (avoids Maven resolution)\n",
            "jars_dir   = PROJECT_ROOT / 'jars'\n",
            "delta_jars = ','.join([p.as_uri() for p in jars_dir.glob('*.jar')])\n",
            "\n",
            "spark = (\n",
            "    SparkSession.builder\n",
            '    .appName("dl-advanced-risk")\n',
            '    .master("local[*]")\n',
            '    .config("spark.jars", delta_jars)\n',
            '    .config("spark.sql.shuffle.partitions", "16")\n',
            '    .config("spark.sql.adaptive.enabled", "true")\n',
            '    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")\n',
            '    .config("spark.sql.catalog.spark_catalog",\n',
            '            "org.apache.spark.sql.delta.catalog.DeltaCatalog")\n',
            "    .getOrCreate()\n",
            ")\n",
            'spark.sparkContext.setLogLevel("WARN")\n',
            'print("Spark version:", spark.version)\n',
        ]
        print("  Fixed Spark session cell (aa000004)")

    # Fix Optuna/MLflow cell - use SQLite backend
    if cid == "aa000010":
        cell["source"] = [
            "# ===========================================================\n",
            "# SECTION 9 \u2013 Optuna hyperparameter search + MLflow\n",
            "# ===========================================================\n",
            "import optuna, mlflow\n",
            'os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"\n',
            "optuna.logging.set_verbosity(optuna.logging.WARNING)\n",
            "\n",
            'mlflow.set_tracking_uri("sqlite:///" + str(PROJECT_ROOT / "mlflow.db"))\n',
            'mlflow.set_experiment("yahoo_finance_risk_dl")\n',
            "\n",
            "class SmallLSTM(nn.Module):\n",
            "    def __init__(self, input_dim, hidden, layers):\n",
            "        super().__init__()\n",
            "        drop = 0.1 if layers > 1 else 0.0\n",
            "        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers,\n",
            "                            batch_first=True, dropout=drop)\n",
            "        self.fc   = nn.Linear(hidden, 1)\n",
            "    def forward(self, x):\n",
            "        out, _ = self.lstm(x)\n",
            "        return self.fc(out[:, -1, :]).squeeze(-1)\n",
            "\n",
            "def objective(trial):\n",
            '    hidden = trial.suggest_categorical("hidden", [32, 64, 128])\n',
            '    lr     = trial.suggest_float("lr", 1e-4, 1e-2, log=True)\n',
            '    layers = trial.suggest_int("layers", 1, 3)\n',
            '    epochs = trial.suggest_int("epochs", 8, 20)\n',
            "\n",
            "    model  = SmallLSTM(INPUT_DIM, hidden, layers)\n",
            "    dl     = make_loader(X_train, y_train, batch_size=512)\n",
            "    _, rmse = train_model(model, dl, X_val, y_val, epochs=epochs, lr=lr)\n",
            "\n",
            '    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):\n',
            "        mlflow.log_params(trial.params)\n",
            '        mlflow.log_metric("val_rmse", rmse)\n',
            "\n",
            "    return rmse\n",
            "\n",
            'with mlflow.start_run(run_name="optuna_search"):\n',
            '    study = optuna.create_study(direction="minimize")\n',
            "    study.optimize(objective, n_trials=15, show_progress_bar=True)\n",
            "    best = study.best_params\n",
            '    mlflow.log_params({"best_" + k: v for k, v in best.items()})\n',
            '    mlflow.log_metric("best_val_rmse", study.best_value)\n',
            "\n",
            'print("\\nBest params:", best)\n',
            'print("Best RMSE  :", study.best_value)\n',
            "\n",
            "try:\n",
            "    ax = optuna.visualization.matplotlib.plot_param_importances(study)\n",
            "    ax.figure.tight_layout()\n",
            '    ax.figure.savefig(str(REPORT_DIR / "optuna_param_importance.png"), dpi=150)\n',
            '    print("Param-importance plot saved.")\n',
            "except Exception as e:\n",
            '    print("Could not plot param importance:", e)\n',
        ]
        print("  Fixed Optuna/MLflow cell (aa000010)")

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Notebook saved successfully!")

