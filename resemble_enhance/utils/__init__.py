from .distributed import global_leader_only
from .logging import setup_logging
from .train_loop import TrainLoop, is_global_leader
from .utils import save_mels, tree_map


def __getattr__(name):
    # The engine wraps DeepSpeed, which only training needs. Load it on first use.
    if name in ("Engine", "gather_attribute"):
        from . import engine

        return getattr(engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
