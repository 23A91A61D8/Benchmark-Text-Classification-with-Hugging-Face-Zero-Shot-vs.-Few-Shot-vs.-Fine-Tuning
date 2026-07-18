"""Entry point: runs zero-shot, few-shot, and fine-tuning benchmarks end to
end and writes the consolidated results/results.json file."""
import json
import os
import random
import time

import numpy as np
import torch

from src import config, zero_shot, few_shot, fine_tune

logger = config.get_logger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(config.SEED)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    logger.info("Device: %s", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting full benchmark pipeline ...")
    pipeline_start = time.perf_counter()

    results = {}

    logger.info("Step 1/3: zero-shot classification")
    results["zero_shot"] = zero_shot.run()

    logger.info("Step 2/3: few-shot classification (SetFit)")
    results["few_shot"] = few_shot.run()

    logger.info("Step 3/3: fine-tuning")
    results["fine_tuned"] = fine_tune.run()

    out_path = os.path.join(config.OUTPUT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    elapsed_min = (time.perf_counter() - pipeline_start) / 60.0
    logger.info("Pipeline complete in %.1f minutes. Results written to %s", elapsed_min, out_path)


if __name__ == "__main__":
    main()
