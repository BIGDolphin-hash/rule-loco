"""Independent post-rule memory modules.

Each child package owns its implementation and may depend only on the neutral
pipeline contracts under :mod:`pro_innovation`.  Child packages must never
import one another.
"""

__all__ = ("global_patch", "gpstm", "gcda", "gpojm")
