"""
Custom exceptions for the TradeSense data layer.

Provides specific, distinguishable exceptions for different failure modes.
"""


class TradeSenseDataError(Exception):
    """Base exception for all TradeSense data layer errors."""

    pass


class ValidationError(TradeSenseDataError):
    """
    Raised when candle or dataset validation fails.

    Attributes:
        message: Human-readable error description
        field: The field that failed validation (if applicable)
        value: The invalid value (if applicable)
    """

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: any = None,
    ):
        super().__init__(message)
        self.message = message
        self.field = field
        self.value = value

    def __str__(self) -> str:
        if self.field and self.value is not None:
            return f"{self.message} (field: {self.field}, value: {self.value})"
        return self.message


class DataProviderError(TradeSenseDataError):
    """
    Raised when an external data provider fails.

    Attributes:
        message: Human-readable error description
        provider: Name of the data provider
        original_error: The underlying exception (if any)
    """

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        original_error: Exception | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.original_error = original_error

    def __str__(self) -> str:
        parts = [self.message]
        if self.provider:
            parts.append(f"provider: {self.provider}")
        return " | ".join(parts)


class ConfigurationError(TradeSenseDataError):
    """
    Raised when configuration is missing or invalid.

    Attributes:
        message: Human-readable error description
        config_key: The missing/invalid configuration key
    """

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.config_key = config_key

    def __str__(self) -> str:
        if self.config_key:
            return f"{self.message} (config_key: {self.config_key})"
        return self.message


class DataQualityError(TradeSenseDataError):
    """
    Raised when dataset-level quality issues are detected.

    Used for issues that span multiple candles, such as:
    - Duplicate timestamps
    - Unsorted data
    - Gaps in the time series

    Attributes:
        message: Human-readable error description
        issues: List of specific issues found
    """

    def __init__(
        self,
        message: str,
        issues: list[str] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.issues = issues or []

    def __str__(self) -> str:
        if self.issues:
            issues_str = "; ".join(self.issues[:5])
            if len(self.issues) > 5:
                issues_str += f" (and {len(self.issues) - 5} more)"
            return f"{self.message}: {issues_str}"
        return self.message
