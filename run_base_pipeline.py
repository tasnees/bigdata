import os, sys, time
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import yfinance as yf

# Windows JVM and Hadoop configuration
JAVA_HOME = r"C:\Program Files\Microsoft\jdk-17.0.21.21-hotspot"
HADOOP_HOME = r"C:\Users\21625\Desktop\bigdata\hadoop"
SPARK_TMP = r"C:\Users\21625\Desktop\bigdata\spark_temp"

os.environ["JAVA_HOME"] = JAVA_HOME
os.environ["HADOOP_HOME"] = HADOOP_HOME
os.environ["SPARK_LOCAL_DIRS"] = SPARK_TMP
hadoop_fwd = HADOOP_HOME.replace("\\", "/")
os.environ["PYSPARK_SUBMIT_ARGS"] = f'--driver-java-options "-Dhadoop.home.dir={hadoop_fwd}" pyspark-shell'

from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.sql.window import Window
from delta import configure_spark_with_delta_pip

PROJECT_ROOT = Path.cwd()
RAW_DIR = PROJECT_ROOT / "data/raw/yahoo_finance"
BRONZE_PATH = PROJECT_ROOT / "data/lakehouse/bronze/market_bars"
SILVER_PATH = PROJECT_ROOT / "data/lakehouse/silver/market_bars"
GOLD_PATH = PROJECT_ROOT / "data/lakehouse/gold/investment_risk_features"
QUARANTINE_PATH = PROJECT_ROOT / "data/lakehouse/quarantine/market_bars"
REPORT_DIR = PROJECT_ROOT / "data/reports"

for p in [RAW_DIR, BRONZE_PATH.parent, SILVER_PATH.parent, GOLD_PATH.parent, QUARANTINE_PATH.parent, REPORT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

DEFAULT_UNIVERSE = [
    ("AAPL", "STOCK"), ("MSFT", "STOCK"), ("NVDA", "STOCK"),
    ("AMZN", "STOCK"), ("GOOGL", "STOCK"), ("META", "STOCK"),
    ("SPY", "ETF"), ("QQQ", "ETF"), ("^GSPC", "INDEX")
]

universe = pd.DataFrame(DEFAULT_UNIVERSE, columns=["symbol", "asset_type"])
print("1. Universe loaded:", len(universe), "tickers")

def batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]

def normalize_download(data, ticker):
    if data is None or data.empty:
        return pd.DataFrame()
    df = data.copy()
    if isinstance(df.columns, pd.MultiIndex):
        level0 = set(map(str, df.columns.get_level_values(0)))
        level1 = set(map(str, df.columns.get_level_values(1)))
        if ticker in level0:
            df = df[ticker]
        elif ticker in level1:
            df = df.xs(ticker, axis=1, level=1)
        else:
            return pd.DataFrame()
    df.columns = [str(c).strip() for c in df.columns]
    needed = {"Open", "High", "Low", "Close", "Volume"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()
    df = df.reset_index()
    date_col = "Date" if "Date" in df.columns else ("Datetime" if "Datetime" in df.columns else df.columns[0])
    # Convert dates to string (YYYY-MM-DD) to prevent INT64 NANOS timestamp incompatibility in Spark Parquet reader
    dates = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
    out = pd.DataFrame({
        "symbol": ticker,
        "observation_date": dates,
        "open": pd.to_numeric(df["Open"], errors="coerce").astype(float),
        "high": pd.to_numeric(df["High"], errors="coerce").astype(float),
        "low": pd.to_numeric(df["Low"], errors="coerce").astype(float),
        "close": pd.to_numeric(df["Close"], errors="coerce").astype(float),
        "volume": pd.to_numeric(df["Volume"], errors="coerce").astype(float)
    })
    out["adjusted_close"] = out["close"]
    out["currency"] = None
    out["source"] = "YAHOO_FINANCE_YFINANCE"
    out["price_basis"] = "AUTO_ADJUSTED_OHLC"
    out["ingestion_timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return out.dropna(subset=["observation_date"])

print("2. Downloading market data from Yahoo Finance...")
for f in RAW_DIR.glob("batch_*.parquet"):
    f.unlink()

ticker_list = universe.symbol.tolist()
for batch_no, ticker_batch in enumerate(batches(ticker_list, 5), start=1):
    data = yf.download(
        tickers=ticker_batch, period="2y", interval="1d",
        auto_adjust=True, progress=False, group_by="ticker", timeout=30
    )
    frames = []
    for ticker in ticker_batch:
        one = normalize_download(data, ticker)
        if not one.empty:
            frames.append(one)
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        # Use pyarrow with coerced timestamps for Spark compatibility
        combined.to_parquet(RAW_DIR / f"batch_{batch_no:05d}.parquet", index=False, engine="pyarrow")
    time.sleep(1)

print("3. Initializing Spark with Delta Lake...")
builder = (
    SparkSession.builder
    .appName("YahooFinance-Pipeline")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.parquet.enableVectorizedReader", "false")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
)
spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.sparkContext.setLogLevel("WARN")

parquet_paths = [str(p) for p in sorted(RAW_DIR.glob("batch_*.parquet"))]
downloaded = spark.read.parquet(*parquet_paths)

universe_schema = T.StructType([
    T.StructField("symbol", T.StringType(), False),
    T.StructField("asset_type", T.StringType(), False)
])
universe_spark = spark.createDataFrame(
    [(r.symbol, r.asset_type) for r in universe.itertuples(index=False)],
    schema=universe_schema
)

bars = (
    downloaded.join(universe_spark, "symbol", "inner")
    .select("symbol", "asset_type", "observation_date", "open", "high", "low", "close", "adjusted_close", "volume", "currency", "source", "price_basis", "ingestion_timestamp")
    .withColumn("symbol", F.upper(F.trim("symbol")))
    .withColumn("observation_date", F.to_date("observation_date"))
)

# Deduplication & Quality check
dedupe_w = Window.partitionBy("symbol", "observation_date").orderBy(F.col("ingestion_timestamp").desc_nulls_last())
deduped = bars.withColumn("_rn", F.row_number().over(dedupe_w)).filter(F.col("_rn") == 1).drop("_rn")

invalid = (
    F.col("symbol").isNull()
    | F.col("observation_date").isNull()
    | F.col("close").isNull()
    | (F.col("close") <= 0)
    | (F.col("high") < F.col("low"))
)
silver = deduped.filter(~invalid)

def write_delta(df, path):
    (df.withColumn("year", F.year("observation_date"))
       .withColumn("month", F.month("observation_date"))
       .write.format("delta").mode("overwrite")
       .partitionBy("year", "month").save(str(path)))

print("4. Writing Bronze & Silver Delta tables...")
write_delta(deduped, BRONZE_PATH)
write_delta(silver, SILVER_PATH)

# Classical Gold features
asset_w = Window.partitionBy("symbol").orderBy("observation_date")
with_returns = silver.withColumn("prev_close", F.lag("close").over(asset_w))\
                     .withColumn("daily_return", F.col("close") / F.col("prev_close") - 1.0)

w20 = Window.partitionBy("symbol").orderBy("observation_date").rowsBetween(-19, 0)
w60 = Window.partitionBy("symbol").orderBy("observation_date").rowsBetween(-59, 0)

gold = with_returns.withColumn("volatility_20d", F.stddev_samp("daily_return").over(w20) * F.sqrt(F.lit(252.0)))\
                   .withColumn("volatility_60d", F.stddev_samp("daily_return").over(w60) * F.sqrt(F.lit(252.0)))\
                   .withColumn("rolling_max_20d", F.max("close").over(w20))\
                   .withColumn("drawdown_20d", F.col("close") / F.col("rolling_max_20d") - 1.0)

write_delta(gold, GOLD_PATH)
print(f"5. Pipeline SUCCESS! Silver bars: {silver.count()} | Gold features: {gold.count()}")
spark.stop()
