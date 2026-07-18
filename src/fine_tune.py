"""Full fine-tuning benchmark: trains a transformer classifier on AG News,
evaluates it, benchmarks inference latency at multiple batch sizes, and
runs a data-efficiency sweep over training-set size."""
import time

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from src import config, data_utils, benchmark_utils

logger = config.get_logger(__name__)

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(config.FINE_TUNE_MODEL)
    return _tokenizer


def _tokenize(dataset):
    tokenizer = _get_tokenizer()

    def _fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=config.FINE_TUNE_MAX_LENGTH,
        )

    return dataset.map(_fn, batched=True, remove_columns=["text"])


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy, macro_f1 = benchmark_utils.accuracy_and_macro_f1(labels, preds)
    return {"accuracy": accuracy, "macro_f1": macro_f1}


def _train_model(train_ds, epochs, output_dir):
    tokenizer = _get_tokenizer()
    model = AutoModelForSequenceClassification.from_pretrained(
        config.FINE_TUNE_MODEL, num_labels=config.NUM_LABELS
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=config.FINE_TUNE_BATCH_SIZE,
        per_device_eval_batch_size=config.FINE_TUNE_BATCH_SIZE,
        learning_rate=config.FINE_TUNE_LR,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        seed=config.SEED,
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
    )
    start = time.perf_counter()
    trainer.train()
    elapsed_minutes = (time.perf_counter() - start) / 60.0
    return model, trainer, elapsed_minutes


def _evaluate_model(trainer, eval_ds):
    metrics = trainer.evaluate(eval_dataset=eval_ds)
    return metrics["eval_accuracy"], metrics["eval_macro_f1"]


def _benchmark_inference_latency(model, texts, batch_size, device):
    tokenizer = _get_tokenizer()
    model.eval()
    batch_texts = (texts * ((batch_size // len(texts)) + 1))[:batch_size]

    def _run():
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            max_length=config.FINE_TUNE_MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            model(**inputs)

    return benchmark_utils.measure_latency(_run)


def run():
    logger.info("=== Fine-tuning: %s ===", config.FINE_TUNE_MODEL)
    device = benchmark_utils.get_device()

    train_ds_raw = data_utils.get_train_subset(config.FINE_TUNE_TRAIN_SIZE)
    eval_ds_raw = data_utils.get_eval_subset(config.FINE_TUNE_EVAL_SIZE)
    train_ds = _tokenize(train_ds_raw)
    eval_ds = _tokenize(eval_ds_raw)

    # --- Main training run -------------------------------------------------
    with benchmark_utils.track_peak_memory() as train_peak:
        model, trainer, training_time_minutes = _train_model(
            train_ds, config.FINE_TUNE_EPOCHS, output_dir="/tmp/ft_main"
        )
    accuracy, macro_f1 = _evaluate_model(trainer, eval_ds)

    # --- Inference latency benchmarks --------------------------------------
    model.to(device)
    sample_texts = eval_ds_raw["text"][:128]

    with benchmark_utils.track_peak_memory() as infer_peak:
        median_bs1, p99_bs1 = _benchmark_inference_latency(model, sample_texts, 1, device)
        median_bs128, p99_bs128 = _benchmark_inference_latency(model, sample_texts, 128, device)

    # --- Data efficiency sweep ----------------------------------------------
    logger.info("Running data-efficiency sweep over sizes: %s", config.DATA_EFFICIENCY_SIZES)
    data_efficiency = {}
    for n in config.DATA_EFFICIENCY_SIZES:
        if n == config.FINE_TUNE_TRAIN_SIZE:
            # Reuse the main run's accuracy to avoid retraining an identical model.
            data_efficiency[f"n_{n}"] = {"accuracy": accuracy}
            continue

        logger.info("Data efficiency: n=%d", n)
        subset_raw = data_utils.get_train_subset(n, seed=config.SEED + n)
        subset = _tokenize(subset_raw)
        _, sub_trainer, _ = _train_model(
            subset, config.DATA_EFFICIENCY_EPOCHS, output_dir=f"/tmp/ft_n{n}"
        )
        sub_accuracy, _ = _evaluate_model(sub_trainer, eval_ds)
        data_efficiency[f"n_{n}"] = {"accuracy": sub_accuracy}
        logger.info("n=%d accuracy=%.4f", n, sub_accuracy)

    result = {
        "training_time_minutes": training_time_minutes,
        "peak_training_gpu_memory_mb": train_peak["mb"],
        "evaluation": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        },
        "inference_bs_1": {
            "median_latency_ms": median_bs1,
            "p99_latency_ms": p99_bs1,
        },
        "inference_bs_128": {
            "median_latency_ms": median_bs128,
            "p99_latency_ms": p99_bs128,
        },
        "peak_inference_gpu_memory_mb": infer_peak["mb"],
        "data_efficiency": data_efficiency,
    }
    logger.info("Fine-tuned results: %s", result)
    return result
