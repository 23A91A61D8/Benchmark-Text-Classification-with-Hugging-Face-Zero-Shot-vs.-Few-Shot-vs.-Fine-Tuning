"""Zero-shot text classification benchmark using a pretrained NLI model."""
from transformers import pipeline

from src import config, data_utils, benchmark_utils

logger = config.get_logger(__name__)


def run():
    logger.info("=== Zero-shot classification: %s ===", config.ZERO_SHOT_MODEL)
    device = 0 if benchmark_utils.get_device() == "cuda" else -1

    classifier = pipeline(
        "zero-shot-classification",
        model=config.ZERO_SHOT_MODEL,
        device=device,
    )
    candidate_labels = data_utils.label_names()

    eval_ds = data_utils.get_eval_subset(config.ZERO_SHOT_EVAL_SIZE)
    texts = eval_ds["text"]
    y_true = eval_ds["label"]

    y_pred = []
    with benchmark_utils.track_peak_memory() as peak:
        for text in texts:
            result = classifier(text, candidate_labels)
            top_label = result["labels"][0]
            y_pred.append(candidate_labels.index(top_label))

    accuracy, macro_f1 = benchmark_utils.accuracy_and_macro_f1(y_true, y_pred)

    sample_text = texts[0]
    median_ms, p99_ms = benchmark_utils.measure_latency(
        lambda: classifier(sample_text, candidate_labels)
    )

    result = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "median_latency_ms": median_ms,
        "p99_latency_ms": p99_ms,
        "peak_gpu_memory_mb": peak["mb"],
    }
    logger.info("Zero-shot results: %s", result)
    return result
