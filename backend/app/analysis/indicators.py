def calculate_sma(
    prices: list[float],
    period: int,
) -> float:
    if len(prices) < period:
        raise ValueError(
            "Not enough price data for the requested period"
        )

    return sum(prices[-period:]) / period


def calculate_ema(
    prices: list[float],
    period: int,
) -> float:
    if len(prices) < period:
        raise ValueError(
            "Not enough price data for the requested period"
        )

    multiplier = 2 / (period + 1)

    ema = sum(prices[:period]) / period

    for price in prices[period:]:
        ema = (
            price * multiplier
            + ema * (1 - multiplier)
        )

    return ema


def calculate_rsi(
    prices: list[float],
    period: int = 14,
) -> float:
    if len(prices) < period + 1:
        raise ValueError(
            "Not enough price data for RSI calculation"
        )

    gains = []
    losses = []

    for i in range(1, len(prices)):
        change = prices[i] - prices[i - 1]

        if change > 0:
            gains.append(change)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(change))

    average_gain = sum(gains[:period]) / period
    average_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        average_gain = (
            (average_gain * (period - 1))
            + gains[i]
        ) / period

        average_loss = (
            (average_loss * (period - 1))
            + losses[i]
        ) / period

    if average_loss == 0:
        return 100.0

    relative_strength = (
        average_gain / average_loss
    )

    return 100 - (
        100 / (1 + relative_strength)
    )


def calculate_macd(
    prices: list[float],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[float, float, float]:

    if len(prices) < slow_period + signal_period:
        raise ValueError(
            "Not enough price data for MACD calculation"
        )

    fast_ema_values = []
    slow_ema_values = []

    for i in range(
        fast_period - 1,
        len(prices),
    ):
        fast_ema_values.append(
            calculate_ema(
                prices[:i + 1],
                fast_period,
            )
        )

    for i in range(
        slow_period - 1,
        len(prices),
    ):
        slow_ema_values.append(
            calculate_ema(
                prices[:i + 1],
                slow_period,
            )
        )

    # Align the fast EMA with the slow EMA
    offset = (
        len(fast_ema_values)
        - len(slow_ema_values)
    )

    fast_ema_values = fast_ema_values[offset:]

    macd_values = [
        fast - slow
        for fast, slow in zip(
            fast_ema_values,
            slow_ema_values,
        )
    ]

    if len(macd_values) < signal_period:
        raise ValueError(
            "Not enough MACD data for signal calculation"
        )

    signal_values = []

    for i in range(
        signal_period - 1,
        len(macd_values),
    ):
        signal_values.append(
            calculate_ema(
                macd_values[:i + 1],
                signal_period,
            )
        )

    macd_value = macd_values[-1]
    signal_value = signal_values[-1]
    histogram_value = (
        macd_value - signal_value
    )

    return (
        macd_value,
        signal_value,
        histogram_value,
    )


def calculate_adx(
    candles,
    period: int = 14,
) -> float:
    if len(candles) < period * 2:
        raise ValueError(
            "Not enough price data for ADX calculation"
        )

    true_ranges = []
    plus_dm = []
    minus_dm = []

    for i in range(1, len(candles)):
        current = candles[i]
        previous = candles[i - 1]

        high = current.high
        low = current.low
        previous_close = previous.close

        true_range = max(
            high - low,
            abs(high - previous_close),
            abs(low - previous_close),
        )

        up_move = high - previous.high
        down_move = previous.low - low

        if up_move > down_move and up_move > 0:
            positive_dm = up_move
        else:
            positive_dm = 0.0

        if down_move > up_move and down_move > 0:
            negative_dm = down_move
        else:
            negative_dm = 0.0

        true_ranges.append(true_range)
        plus_dm.append(positive_dm)
        minus_dm.append(negative_dm)

    if len(true_ranges) < period:
        raise ValueError(
            "Not enough data for ADX calculation"
        )

    atr = sum(
        true_ranges[:period]
    ) / period

    smoothed_plus_dm = sum(
        plus_dm[:period]
    ) / period

    smoothed_minus_dm = sum(
        minus_dm[:period]
    ) / period

    dx_values = []

    for i in range(period, len(true_ranges)):
        atr = (
            (atr * (period - 1))
            + true_ranges[i]
        ) / period

        smoothed_plus_dm = (
            (smoothed_plus_dm * (period - 1))
            + plus_dm[i]
        ) / period

        smoothed_minus_dm = (
            (smoothed_minus_dm * (period - 1))
            + minus_dm[i]
        ) / period

        if atr == 0:
            continue

        plus_di = (
            100
            * smoothed_plus_dm
            / atr
        )

        minus_di = (
            100
            * smoothed_minus_dm
            / atr
        )

        di_sum = plus_di + minus_di

        if di_sum == 0:
            dx = 0.0
        else:
            dx = (
                100
                * abs(plus_di - minus_di)
                / di_sum
            )

        dx_values.append(dx)

    if len(dx_values) < period:
        raise ValueError(
            "Not enough data for ADX calculation"
        )

    adx = sum(
        dx_values[:period]
    ) / period

    for dx in dx_values[period:]:
        adx = (
            (adx * (period - 1))
            + dx
        ) / period

    return adx