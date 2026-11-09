"""
eval_dl_models_part2.py – Optuna HPO, SHAP, Flask serving, Spark UDF.
Picks up after the three models already trained.
"""
import os, sys, pathlib, time, json, threading
import warnings
warnings.filterwarnings("ignore")

# ── 0. Environment ──────────────────────────────────────────────────────
HADOOP_HOME = r"C:\Users\21625\Desktop\bigdata\hadoop"
JAVA_HOME   = r"C:\Program Files\Microsoft\jdk-17.0.21.21-hotspot"
SPARK_TMP   = r"C:\Users\21625\Desktop\bigdata\spark_temp"
PROJECT_ROOT = pathlib.Path(r"C:\Users\21625\Desktop\bigdata")

os.environ["HADOOP_HOME"]      = HADOOP_HOME
os.environ["JAVA_HOME"]        = JAVA_HOME
os.environ["SPARK_LOCAL_DIRS"] = SPARK_TMP
os.environ["PATH"] = HADOOP_HOME + r"\bin;" + os.environ.get("PATH", "")
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
hadoop_fwd = HADOOP_HOME.replace("\\", "/")
os.environ["PYSPARK_SUBMIT_ARGS"] = (
    f'--driver-java-options "-Dhadoop.home.dir={hadoop_fwd}" pyspark-shell'
)

MODEL_DIR  = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Spark session ────────────────────────────────────────────────────
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

jars_dir   = PROJECT_ROOT / "jars"
delta_jars = ",".join([p.as_uri() for p in jars_dir.glob("*.jar")])


os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["PYSPARK_GATEWAY_SECRET"] = "antigravity_secret"


spark = (
    SparkSession.builder
    .appName("dl-risk-eval-part2")
    .master("local[*]")
    .config("spark.jars", delta_jars)
    .config("spark.sql.shuffle.partitions", "16")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

# ── 2. Rebuild features (fast) ──────────────────────────────────────────
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SILVER_PATH = str(PROJECT_ROOT / "data" / "lakehouse" / "silver" / "market_bars")
pdf = spark.read.format("delta").load(SILVER_PATH).toPandas()
pdf = pdf.sort_values(["symbol", "observation_date"]).reset_index(drop=True)
close_col = "close"

pdf["log_ret"] = np.log(pdf[close_col] / pdf[close_col].shift(1))
pdf["vol_10"]  = pdf.groupby("symbol")["log_ret"].transform(
    lambda s: s.rolling(10, min_periods=1).std()
)

def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_macd_signal(series, fast=12, slow=26, signal=9):
    ema_fast  = series.ewm(span=fast, adjust=False).mean()
    ema_slow  = series.ewm(span=slow, adjust=False).mean()
    return (ema_fast - ema_slow).ewm(span=signal, adjust=False).mean()

def compute_bb_width(series, period=20):
    sma = series.rolling(period, min_periods=1).mean()
    std = series.rolling(period, min_periods=1).std().replace(0, np.nan)
    return (2 * std) / sma

pdf["rsi_14"]      = pdf.groupby("symbol")[close_col].transform(compute_rsi)
pdf["macd_signal"] = pdf.groupby("symbol")[close_col].transform(compute_macd_signal)
pdf["bb_width"]    = pdf.groupby("symbol")[close_col].transform(compute_bb_width)
pdf["target"]      = pdf.groupby("symbol")["vol_10"].shift(-1).fillna(0)

feature_cols = ["log_ret", "vol_10", "rsi_14", "macd_signal", "bb_width"]
pdf[feature_cols] = pdf[feature_cols].fillna(0)

symbols  = sorted(pdf["symbol"].unique())
sym2idx  = {s: i for i, s in enumerate(symbols)}
num_syms = len(symbols)
pdf["sym_idx"] = pdf["symbol"].map(sym2idx)

split_date = pd.Timestamp("2025-01-01")
train_mask = pd.to_datetime(pdf["observation_date"]) < split_date

X      = pdf[feature_cols].values.astype(np.float32)
y      = pdf["target"].values.astype(np.float32)
sym_id = pdf["sym_idx"].values.astype(np.int64)

X_train, X_val = X[train_mask], X[~train_mask]
y_train, y_val = y[train_mask], y[~train_mask]
s_train, s_val = sym_id[train_mask], sym_id[~train_mask]

INPUT_DIM = len(feature_cols)
print(f"Data ready — Train: {X_train.shape[0]}  Val: {X_val.shape[0]}")

# ── Model definitions (needed for saving/loading) ───────────────────────
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.relu = nn.ReLU(); self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self._pad = pad
    def forward(self, x):
        out = self.conv(x)
        if self._pad > 0: out = out[:, :, :-self._pad]
        return self.dropout(self.relu(out)) + self.residual(x)

class VolatilityTCN(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, channels=32):
        super().__init__()
        self.net = nn.Sequential(
            TCNBlock(input_dim, channels, dilation=1),
            TCNBlock(channels, channels, dilation=2),
            TCNBlock(channels, channels, dilation=4),
        )
        self.fc = nn.Linear(channels, 1)
    def forward(self, x):
        out = self.net(x.permute(0, 2, 1))
        return self.fc(out[:, :, -1]).squeeze(-1)

class VolatilityTransformer(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64,
            batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
    def forward(self, x):
        return self.fc(self.encoder(self.proj(x))[:, -1, :]).squeeze(-1)

class MultiAssetLSTM(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden=32, layers=2,
                 num_symbols=50, embed_dim=8):
        super().__init__()
        self.embed = nn.Embedding(num_symbols, embed_dim)
        self.lstm = nn.LSTM(input_dim + embed_dim, hidden,
                            num_layers=layers, batch_first=True,
                            dropout=0.1 if layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x, sym_idx):
        emb = self.embed(sym_idx).unsqueeze(1).expand(-1, x.size(1), -1)
        out, _ = self.lstm(torch.cat([x, emb], dim=-1))
        return self.fc(out[:, -1, :]).squeeze(-1)

# ── Training utility ────────────────────────────────────────────────────
def make_loader(X, y, sym=None, batch_size=512, shuffle=True):
    tensors = [torch.from_numpy(X).unsqueeze(1), torch.from_numpy(y)]
    if sym is not None: tensors.insert(1, torch.from_numpy(sym))
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)

def train_model(model, train_dl, val_X, val_y, val_sym=None,
                epochs=15, lr=1e-3, dev=device, verbose=False):
    model = model.to(dev)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_rmse = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_dl:
            if len(batch) == 3:
                xb, symb, yb = [t.to(dev) for t in batch]
                pred = model(xb, symb)
            else:
                xb, yb = [t.to(dev) for t in batch]
                pred = model(xb)
            optimizer.zero_grad()
            nn.MSELoss()(pred, yb).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(val_X).unsqueeze(1).to(dev)
            vy = torch.from_numpy(val_y).to(dev)
            vp = model(vx, torch.from_numpy(val_sym).to(dev)) if val_sym is not None else model(vx)
            rmse = torch.sqrt(criterion(vp, vy)).item()
        best_rmse = min(best_rmse, rmse)
    return model, best_rmse

# ── Re-train TCN (quick, for saving) ────────────────────────────────────
print("\nRe-training TCN (15 epochs)...")
tcn_model, tcn_rmse = train_model(
    VolatilityTCN(INPUT_DIM), make_loader(X_train, y_train),
    X_val, y_val, epochs=15, verbose=False)
print(f"  TCN best val RMSE: {tcn_rmse:.6f}")

print("Re-training Transformer (15 epochs)...")
tf_model, tf_rmse = train_model(
    VolatilityTransformer(INPUT_DIM), make_loader(X_train, y_train),
    X_val, y_val, epochs=15, verbose=False)
print(f"  Transformer best val RMSE: {tf_rmse:.6f}")

print("Re-training MultiAssetLSTM (15 epochs)...")
ma_model, ma_rmse = train_model(
    MultiAssetLSTM(INPUT_DIM, num_symbols=num_syms),
    make_loader(X_train, y_train, sym=s_train),
    X_val, y_val, val_sym=s_val, epochs=15, verbose=False)
print(f"  MultiAssetLSTM best val RMSE: {ma_rmse:.6f}")

# Save checkpoints
torch.save(tcn_model.state_dict(), str(MODEL_DIR / "vol_tcn.pt"))
torch.save(tf_model.state_dict(),  str(MODEL_DIR / "vol_transformer.pt"))
torch.save(ma_model.state_dict(),  str(MODEL_DIR / "vol_multi_asset_lstm.pt"))
print("Checkpoints saved to:", MODEL_DIR)

# ── 3. Optuna HPO + MLflow ──────────────────────────────────────────────
import optuna
import mlflow
optuna.logging.set_verbosity(optuna.logging.WARNING)

mlflow.set_tracking_uri("sqlite:///" + str(PROJECT_ROOT / "mlflow.db"))
mlflow.set_experiment("yahoo_finance_risk_dl")

class SmallLSTM(nn.Module):
    def __init__(self, input_dim, hidden, layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers,
                            batch_first=True, dropout=0.1 if layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

def objective(trial):
    hidden = trial.suggest_categorical("hidden", [32, 64, 128])
    lr     = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    layers = trial.suggest_int("layers", 1, 3)
    epochs = trial.suggest_int("epochs", 8, 20)
    model  = SmallLSTM(INPUT_DIM, hidden, layers)
    dl     = make_loader(X_train, y_train, batch_size=512)
    _, rmse = train_model(model, dl, X_val, y_val, epochs=epochs, lr=lr)
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        mlflow.log_params(trial.params)
        mlflow.log_metric("val_rmse", rmse)
    return rmse

print("\n" + "="*60)
print("OPTUNA HYPERPARAMETER SEARCH (15 trials)")
print("="*60)

with mlflow.start_run(run_name="optuna_search"):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15)
    best = study.best_params
    mlflow.log_params({"best_" + k: v for k, v in best.items()})
    mlflow.log_metric("best_val_rmse", study.best_value)

print(f"\nBest params: {best}")
print(f"Best RMSE  : {study.best_value:.6f}")

# Log best models to MLflow
with mlflow.start_run(run_name="best_tcn"):
    mlflow.log_metric("val_rmse", tcn_rmse)
with mlflow.start_run(run_name="best_transformer"):
    mlflow.log_metric("val_rmse", tf_rmse)
with mlflow.start_run(run_name="best_multi_asset_lstm"):
    mlflow.log_metric("val_rmse", ma_rmse)

try:
    import matplotlib
    matplotlib.use("Agg")
    ax = optuna.visualization.matplotlib.plot_param_importances(study)
    ax.figure.tight_layout()
    ax.figure.savefig(str(REPORT_DIR / "optuna_param_importance.png"), dpi=150)
    print("Param-importance plot saved.")
except Exception as e:
    print(f"Could not plot param importance: {e}")

# ── 4. SHAP explainability ──────────────────────────────────────────────
print("\n" + "="*60)
print("SHAP EXPLAINABILITY (KernelExplainer on TCN)")
print("="*60)
import shap
import matplotlib.pyplot as plt

explain_model = tcn_model.cpu().eval()

def model_predict_numpy(x_np):
    with torch.no_grad():
        t = torch.from_numpy(x_np.astype(np.float32)).unsqueeze(1)
        return explain_model(t).numpy()

bg_size  = min(100, len(X_train))
val_size = min(50,  len(X_val))

print("Running SHAP KernelExplainer...")
explainer = shap.KernelExplainer(model_predict_numpy, X_train[:bg_size])
shap_vals = explainer.shap_values(X_val[:val_size], nsamples=100)

plt.figure(figsize=(8, 4))
shap.summary_plot(shap_vals, X_val[:val_size],
                  feature_names=feature_cols, show=False)
plt.tight_layout()
shap_path = str(REPORT_DIR / "shap_summary.png")
plt.savefig(shap_path, dpi=150)
print("SHAP summary saved to:", shap_path)

# ── 5. Flask REST server + Spark UDF ────────────────────────────────────
print("\n" + "="*60)
print("FLASK MODEL SERVING + SPARK UDF INFERENCE")
print("="*60)
import flask
import requests as req

serve_app = flask.Flask("risk_model_server")
_cache = {}

def _get_model():
    if "tcn" not in _cache:
        m = VolatilityTCN(INPUT_DIM)
        m.load_state_dict(torch.load(str(MODEL_DIR / "vol_tcn.pt"),
                                     map_location="cpu", weights_only=False))
        m.eval()
        _cache["tcn"] = m
    return _cache["tcn"]

@serve_app.route("/predict", methods=["POST"])
def predict():
    data = flask.request.get_json(force=True)
    features = np.array(data["features"], dtype=np.float32)
    inp = torch.from_numpy(features).unsqueeze(1)
    with torch.no_grad():
        out = _get_model()(inp).squeeze().tolist()
    if isinstance(out, float): out = [out]
    return flask.jsonify({"predicted_vol": out})

@serve_app.route("/health")
def health():
    return flask.jsonify({"status": "ok"})

SERVER_PORT = 5050
t = threading.Thread(
    target=lambda: serve_app.run(port=SERVER_PORT, debug=False, use_reloader=False),
    daemon=True)
t.start()
time.sleep(1.5)

health_resp = req.get(f"http://localhost:{SERVER_PORT}/health", timeout=5)
print("Flask server:", health_resp.json())

sample = X_val[:4].tolist()
resp = req.post(f"http://localhost:{SERVER_PORT}/predict",
                json={"features": sample}, timeout=10)
print("Sample predictions:", resp.json())

# Spark UDF calling Flask endpoint
from pyspark.sql.functions import pandas_udf

@pandas_udf("double")
def dl_risk_score(*cols: pd.Series) -> pd.Series:
    feat = np.stack([c.values for c in cols], axis=1).tolist()
    r = req.post(f"http://localhost:{SERVER_PORT}/predict",
                 json={"features": feat}, timeout=30)
    return pd.Series(r.json()["predicted_vol"])

GOLD_PATH = str(PROJECT_ROOT / "data" / "lakehouse" / "gold" / "investment_risk_features")
gold_df = spark.read.format("delta").load(GOLD_PATH)

if "log_ret" not in gold_df.columns:
    w_sym = Window.partitionBy("symbol").orderBy("observation_date")
    gold_df = gold_df.withColumn(
        "log_ret", F.log(F.col("close") / F.lag("close").over(w_sym))
    ).na.fill({"log_ret": 0.0})

for col_name in ["vol_10", "rsi_14", "macd_signal", "bb_width"]:
    if col_name not in gold_df.columns:
        gold_df = gold_df.withColumn(col_name, F.lit(0.0))

scored = gold_df.withColumn(
    "dl_predicted_vol",
    dl_risk_score("log_ret", "vol_10", "rsi_14", "macd_signal", "bb_width")
)

DL_GOLD = str(PROJECT_ROOT / "data" / "lakehouse" / "gold" / "dl_risk_scores")
scored.write.format("delta").mode("overwrite").save(DL_GOLD)

print("\nDL risk scores written to:", DL_GOLD)
scored.select("symbol", "observation_date", "close", "dl_predicted_vol").show(10)

# ── 6. Final summary ────────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
results = {"TCN": tcn_rmse, "Transformer": tf_rmse, "MultiAssetLSTM": ma_rmse}
print(f"\n{'Model':<25s} {'Val RMSE':>12s}")
print("-" * 38)
for name, rmse in sorted(results.items(), key=lambda kv: kv[1]):
    print(f"  {name:<23s} {rmse:>12.6f}")
print(f"\nOptuna best trial:")
print(f"  Params: {study.best_params}")
print(f"  RMSE  : {study.best_value:.6f}")
print(f"\nArtifacts:")
print(f"  Models     : {MODEL_DIR}")
print(f"  SHAP plot  : {REPORT_DIR / 'shap_summary.png'}")
print(f"  Optuna plot: {REPORT_DIR / 'optuna_param_importance.png'}")
print(f"  MLflow DB  : {PROJECT_ROOT / 'mlflow.db'}")
print(f"  DL scores  : {DL_GOLD}")

spark.stop()
print("\nDone!")
