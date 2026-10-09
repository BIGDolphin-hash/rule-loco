"""Domain-specific exceptions used by the V1 pipeline."""


class ProInnovationError(Exception):
    """Base class for expected project errors."""


class TemplateError(ProInnovationError, ValueError):
    """A rule template is malformed or inconsistent."""


class ExecutionError(ProInnovationError, RuntimeError):
    """A valid rule could not be evaluated."""


class ModelAdapterError(ProInnovationError, RuntimeError):
    """A local model adapter could not be initialized or invoked."""


class MemoryBankError(ProInnovationError, RuntimeError):
    """A patch memory bank is missing, incompatible, or malformed."""
