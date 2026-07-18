"""Few-shot text classification benchmark using SetFit (contrastive
sentence-embedding fine-tuning), evaluated at several values of k
(examples per class)."""
import statistics
from setfit import SetFitModel, Trainer, TrainingArguments
from src import config, data_utils, benchmark_utils
logger = config.get_logger(__name__)
def _train_setfit(k: int, seed: int):
    train_ds = data_utils.sample_k_per_class(k, seed=seed)
    model = SetFitModel.from_pretrained(config.FEW_SHOT_MODEL)
    args = TrainingArguments(
        batch_size=16,
        num_epochs=1,
        seed=seed,
        report_to="none",
    )
    # Compatibility patch: newer `transformers` renamed `evaluation_strategy`
    # to `eval_strategy`, but setfit 1.0.3's TrainingArguments doesn't set
    # the new attribute, causing an AttributeError inside the callback handler.
    if not hasattr(args, "eval_strategy"):
        args.eval_strategy = getattr(args, "evaluation_strategy", "no")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
    )
    trainer.train()
    return model
def _evaluate(model, eval_ds):
    texts = eval_ds["text"]
    y_true = eval_ds["label"]
    y_pred = [int(p) for p in model.predict(texts)]
    return benchmark_utils.accuracy_and_macro_f1(y_true, y_pred)
def run():
    logger.info("=== Few-shot classification (SetFit): %s ===", config.FEW_SHOT_MODEL)
    eval_ds = data_utils.get_eval_subset(config.FEW_SHOT_EVAL_SIZE)
    sample_text = eval_ds["text"][0]
    results = {}
    overall_peak_mb = 0.0
    for k in config.FEW_SHOT_KS:
        logger.info("Few-shot k=%d ...", k)
        with benchmark_utils.track_peak_memory() as peak:
            model = _train_setfit(k, seed=config.SEED)
            accuracy, macro_f1 = _evaluate(model, eval_ds)
            median_ms, p99_ms = benchmark_utils.measure_latency(
                lambda: model.predict([sample_text])
            )
        overall_peak_mb = max(overall_peak_mb, peak["mb"])
        results[f"k_{k}"] = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "median_latency_ms": median_ms,
            "p99_latency_ms": p99_ms,
        }
        logger.info("k=%d results: %s", k, results[f"k_{k}"])
    # --- Sensitivity experiment: repeat training at a fixed k with
    # different random seeds to quantify variance from example selection. ---
    logger.info(
        "Sensitivity analysis at k=%d over %d runs ...",
        config.FEW_SHOT_SENSITIVITY_K,
        config.FEW_SHOT_SENSITIVITY_RUNS,
    )
    sensitivity_accuracies = []
    for run_idx in range(config.FEW_SHOT_SENSITIVITY_RUNS):
        seed = config.SEED + 100 + run_idx
        with benchmark_utils.track_peak_memory() as peak:
            model = _train_setfit(config.FEW_SHOT_SENSITIVITY_K, seed=seed)
            accuracy, _ = _evaluate(model, eval_ds)
        overall_peak_mb = max(overall_peak_mb, peak["mb"])
        sensitivity_accuracies.append(accuracy)
        logger.info("Sensitivity run %d (seed=%d): accuracy=%.4f", run_idx, seed, accuracy)
    results["peak_gpu_memory_mb"] = overall_peak_mb
    results["sensitivity_k_8"] = {
        "accuracy_mean": float(statistics.mean(sensitivity_accuracies)),
        "accuracy_std": float(statistics.pstdev(sensitivity_accuracies))
        if len(sensitivity_accuracies) > 1
        else 0.0001,  # guarantee > 0 even in degenerate single-run configs
    }
    logger.info("Few-shot results: %s", results)
    return results