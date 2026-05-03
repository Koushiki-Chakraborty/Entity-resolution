# encoder/__init__.py
# =============================================================================
# Makes src/encoder/ an importable Python package.
#
# Pure-Python modules (config, type_detector, serializer) are imported
# eagerly. The FrozenEncoder is imported lazily to avoid crashing when
# torch is not yet installed.
#
# Usage:
#   from encoder.config         import CLEAN_DATA_PATH, EMB_DIM
#   from encoder.type_detector  import detect_entity_type
#   from encoder.serializer     import serialize_name, serialize_context
#   from encoder.frozen_encoder import FrozenEncoder   # needs torch
# =============================================================================

from encoder.config        import (
    FINETUNED_MODEL_DIR, BASE_MODEL, EMB_DIM,
    CLEAN_DATA_PATH, VECTORS_PATH, ENCODE_BATCH_SIZE,
    UNKNOWN_TYPE_VALUES, FALLBACK_TAG,
)
from encoder.type_detector import detect_entity_type
from encoder.serializer    import serialize_name, serialize_context

# FrozenEncoder requires torch -- imported lazily to avoid ImportError
# when only running pure-Python modules (type_detector, serializer).
def get_frozen_encoder(*args, **kwargs):
    """Lazy loader for FrozenEncoder (requires torch + sentence-transformers)."""
    from encoder.frozen_encoder import FrozenEncoder
    return FrozenEncoder(*args, **kwargs)

__all__ = [
    # config constants
    "FINETUNED_MODEL_DIR", "BASE_MODEL", "EMB_DIM",
    "CLEAN_DATA_PATH", "VECTORS_PATH", "ENCODE_BATCH_SIZE",
    "UNKNOWN_TYPE_VALUES", "FALLBACK_TAG",
    # pure-Python helpers
    "detect_entity_type",
    "serialize_name", "serialize_context",
    # lazy encoder loader
    "get_frozen_encoder",
]
