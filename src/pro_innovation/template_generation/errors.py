"""Errors at the template-generation task boundary."""

from ..errors import ProInnovationError


class TemplateGenerationError(ProInnovationError, RuntimeError):
    """Generation failed; no unvalidated template pair is returned."""
