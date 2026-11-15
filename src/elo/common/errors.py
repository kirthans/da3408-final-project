"""Domain errors with stable codes for later HTTP exception handlers."""


class EloError(Exception):
    """Base error for Guess the Elo."""

    code: str = "elo_error"
    http_status: int = 500


class InvalidPgnError(EloError):
    """The input cannot be parsed as a valid PGN."""

    code = "invalid_pgn"
    http_status = 400


class PayloadTooLargeError(EloError):
    """The upload exceeds the configured size limit."""

    code = "payload_too_large"
    http_status = 413


class EngineError(EloError):
    """The chess engine failed to complete analysis."""

    code = "engine_error"
    http_status = 503


class EngineTimeoutError(EngineError):
    """The chess engine exceeded its time budget."""

    code = "engine_timeout"
    http_status = 504


class ModelUnavailableError(EloError):
    """The model service cannot currently accept predictions."""

    code = "model_unavailable"
    http_status = 503


class PipelineBusyError(EloError):
    """A pipeline run is already in progress."""

    code = "pipeline_busy"
    http_status = 409


class DataValidationError(EloError):
    """Input data does not satisfy the expected schema or constraints."""

    code = "data_validation"
    http_status = 422
