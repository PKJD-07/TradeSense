from datetime import datetime

from backend.app.signals.schema import SignalAction, TradingSignal


class MLSignalAdapter:
    def __init__(
        self,
        buy_threshold: float = 0.60,
        sell_threshold: float = 0.60,
    ):
        if not 0.0 <= buy_threshold <= 1.0:
            raise ValueError("buy_threshold must be between 0 and 1")

        if not 0.0 <= sell_threshold <= 1.0:
            raise ValueError("sell_threshold must be between 0 and 1")

        if buy_threshold + sell_threshold <= 1.0:
            raise ValueError(
                "buy_threshold + sell_threshold must be greater than 1"
            )

        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def adapt(
        self,
        timestamp: datetime,
        symbol: str,
        probability_down: float,
        probability_neutral: float,
        probability_up: float,
        source: str = "ML",
        model: str = "unknown",
    ) -> TradingSignal:

        probabilities = [
            probability_down,
            probability_neutral,
            probability_up,
        ]

        if any(
            probability < 0.0 or probability > 1.0
            for probability in probabilities
        ):
            raise ValueError(
                "All probabilities must be between 0 and 1"
            )

        probability_sum = sum(probabilities)

        if abs(probability_sum - 1.0) > 1e-6:
            raise ValueError(
                "Signal probabilities must sum to 1.0"
            )

        confidence = max(probabilities)

        if probability_up >= self.buy_threshold:
            action = SignalAction.BUY

        elif probability_down >= self.sell_threshold:
            action = SignalAction.SELL

        else:
            action = SignalAction.HOLD

        return TradingSignal(
            timestamp=timestamp,
            symbol=symbol,
            action=action,
            confidence=confidence,
            source=source,
            model=model,
            probability_down=probability_down,
            probability_neutral=probability_neutral,
            probability_up=probability_up,
        )