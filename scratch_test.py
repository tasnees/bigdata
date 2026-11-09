import os, pathlib
HADOOP_HOME = r"C:\Users\21625\Desktop\bigdata\hadoop"
os.environ["HADOOP_HOME"] = HADOOP_HOME
os.environ["JAVA_HOME"] = r"C:\Program Files\Microsoft\jdk-17.0.21.21-hotspot"
os.environ["PATH"] = HADOOP_HOME + r"\bin;" + os.environ.get("PATH", "")
os.environ["PYSPARK_SUBMIT_ARGS"] = f'--driver-java-options "-Dhadoop.home.dir={HADOOP_HOME.replace(chr(92), "/")}" pyspark-shell'

from pyspark.sql import SparkSession
jars_dir = pathlib.Path(r"C:\Users\21625\Desktop\bigdata\jars")
delta_jars = ",".join([p.as_uri() for p in jars_dir.glob("*.jar")])

spark = (
    SparkSession.builder
    .appName("nativeio-test")
    .master("local[2]")
    .config("spark.jars", delta_jars)
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .getOrCreate()
)
jvm = spark._jvm
try:
    native_io_cls = jvm.java.lang.Class.forName("org.apache.hadoop.io.nativeio.NativeIO$Windows")
    field = native_io_cls.getDeclaredField("nativeLoaded")
    field.setAccessible(True)
    field.setBoolean(None, False)
    print("NativeIO.Windows.nativeLoaded successfully set to False!")
except Exception as e:
    print("Reflection error:", e)

silver_path = r"C:\Users\21625\Desktop\bigdata\data\lakehouse\silver\market_bars"
df = spark.read.format("delta").load(silver_path)
print("Spark loaded Delta successfully! Count:", df.count())
spark.stop()
