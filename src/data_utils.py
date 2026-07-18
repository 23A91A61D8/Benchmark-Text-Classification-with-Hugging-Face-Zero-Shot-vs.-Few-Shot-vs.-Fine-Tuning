"""Dataset loading and sampling helpers built on top of `datasets`."""
import random
from collections import defaultdict

from datasets import load_dataset, Dataset

from src import config

logger = config.get_logger(__name__)

_DATASET_CACHE = {}


def load_ag_news():
    """Load and cache the AG News dataset (train/test splits)."""
    if "ag_news" not in _DATASET_CACHE:
        logger.info("Loading dataset '%s' ...", config.DATASET_NAME)
        ds = load_dataset(config.DATASET_NAME)
        _DATASET_CACHE["ag_news"] = ds
    return _DATASET_CACHE["ag_news"]


def get_eval_subset(n: int, split: str = "test", seed: int = None) -> Dataset:
    """Return a shuffled, stratified-ish subset of `n` examples for evaluation."""
    seed = seed if seed is not None else config.SEED
    ds = load_ag_news()[split]
    n = min(n, len(ds))
    return ds.shuffle(seed=seed).select(range(n))


def get_train_subset(n: int, seed: int = None) -> Dataset:
    """Return a shuffled subset of `n` training examples."""
    seed = seed if seed is not None else config.SEED
    ds = load_ag_news()["train"]
    n = min(n, len(ds))
    return ds.shuffle(seed=seed).select(range(n))


def sample_k_per_class(k: int, seed: int = None, split: str = "train") -> Dataset:
    """Sample exactly `k` examples per class label (for few-shot / SetFit)."""
    seed = seed if seed is not None else config.SEED
    rng = random.Random(seed)
    ds = load_ag_news()[split]

    by_label = defaultdict(list)
    for idx, label in enumerate(ds["label"]):
        by_label[label].append(idx)

    selected_indices = []
    for label, indices in by_label.items():
        rng.shuffle(indices)
        selected_indices.extend(indices[:k])

    rng.shuffle(selected_indices)
    return ds.select(selected_indices)


def label_names():
    return config.LABEL_NAMES
