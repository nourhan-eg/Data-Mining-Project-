from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

RAW_NSL_KDD = PROJECT_ROOT / "data" / "raw" / "nsl_kdd_sample.csv"
RAW_STUDENT = PROJECT_ROOT / "data" / "raw" / "student_performance_10k.csv"
RAW_RETAIL = PROJECT_ROOT / "data" / "raw" / "online_retail_sample.csv"

CLEAN_NSL_KDD = PROJECT_ROOT / "data" / "processed" / "nsl_kdd_clean.csv"
CLEAN_STUDENT = PROJECT_ROOT / "data" / "processed" / "student_clean.csv"
CLEAN_RETAIL_BASKET = PROJECT_ROOT / "data" / "processed" / "retail_basket.csv"

STUDENT_TARGET = "Performance Index"
TEST_SIZE = 0.2
RANDOM_STATE = 42
