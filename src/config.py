"""Central configuration, loaded from environment variables (.env.example)."""
import os
import logging


def _list(name: str, default: str, cast=str):
    raw = os.getenv(name, default)
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


# --- Dataset ---------------------------------------------------------------
DATASET_NAME = os.getenv("DATASET_NAME", "ag_news")
NUM_LABELS = int(os.getenv("NUM_LABELS", 4))
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

# --- Zero-shot ---------------------------------------------------------------
ZERO_SHOT_MODEL = os.getenv("ZERO_SHOT_MODEL", "facebook/bart-large-mnli")
ZERO_SHOT_EVAL_SIZE = int(os.getenv("ZERO_SHOT_EVAL_SIZE", 500))

# --- Few-shot (SetFit) -------------------------------------------------------
FEW_SHOT_MODEL = os.getenv("FEW_SHOT_MODEL", "sentence-transformers/paraphrase-mpnet-base-v2")
FEW_SHOT_KS = _list("FEW_SHOT_KS", "2,4,8,16", int)
FEW_SHOT_EVAL_SIZE = int(os.getenv("FEW_SHOT_EVAL_SIZE", 500))
FEW_SHOT_SENSITIVITY_K = int(os.getenv("FEW_SHOT_SENSITIVITY_K", 8))
FEW_SHOT_SENSITIVITY_RUNS = int(os.getenv("FEW_SHOT_SENSITIVITY_RUNS", 3))

# --- Fine-tuning --------------------------------------------------------------
FINE_TUNE_MODEL = os.getenv("FINE_TUNE_MODEL", "distilbert-base-uncased")
FINE_TUNE_EPOCHS = float(os.getenv("FINE_TUNE_EPOCHS", 2))
FINE_TUNE_BATCH_SIZE = int(os.getenv("FINE_TUNE_BATCH_SIZE", 32))
FINE_TUNE_LR = float(os.getenv("FINE_TUNE_LR", 5e-5))
FINE_TUNE_MAX_LENGTH = int(os.getenv("FINE_TUNE_MAX_LENGTH", 128))
FINE_TUNE_TRAIN_SIZE = int(os.getenv("FINE_TUNE_TRAIN_SIZE", 120000))
FINE_TUNE_EVAL_SIZE = int(os.getenv("FINE_TUNE_EVAL_SIZE", 2000))
DATA_EFFICIENCY_SIZES = _list("DATA_EFFICIENCY_SIZES", "100,500,2000,10000,120000", int)
DATA_EFFICIENCY_EPOCHS = float(os.getenv("DATA_EFFICIENCY_EPOCHS", 3))

# --- Latency benchmarking -----------------------------------------------------
LATENCY_WARMUP_RUNS = int(os.getenv("LATENCY_WARMUP_RUNS", 5))
LATENCY_MEASURED_RUNS = int(os.getenv("LATENCY_MEASURED_RUNS", 50))
LATENCY_BATCH_SIZES = _list("LATENCY_BATCH_SIZES", "1,128", int)

# --- Misc -----------------------------------------------------------------
SEED = int(os.getenv("SEED", 42))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "results")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)
