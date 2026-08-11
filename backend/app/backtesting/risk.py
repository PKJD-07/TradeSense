def calculate_position_size(
    capital: float,
    entry_price: float,
    stop_loss_percent: float = 2.0,
) -> float:
    """
    Calculate how many shares can be purchased
    while risking only the specified percentage
    of available capital.
    """

    if capital <= 0:
        raise ValueError("Capital must be greater than zero")

    if entry_price <= 0:
        raise ValueError("Entry price must be greater than zero")

    if stop_loss_percent <= 0:
        raise ValueError("Stop loss percentage must be greater than zero")

    risk_amount = capital * (stop_loss_percent / 100)

    stop_loss_price = entry_price * (
        1 - stop_loss_percent / 100
    )

    risk_per_share = entry_price - stop_loss_price

    if risk_per_share <= 0:
        raise ValueError("Invalid risk per share")

    position_size = risk_amount / risk_per_share

    return position_size


def calculate_stop_loss(
    entry_price: float,
    stop_loss_percent: float = 2.0,
) -> float:
    """
    Calculate stop-loss price for a long position.
    """

    if entry_price <= 0:
        raise ValueError("Entry price must be greater than zero")

    if stop_loss_percent <= 0:
        raise ValueError("Stop loss percentage must be greater than zero")

    return entry_price * (
        1 - stop_loss_percent / 100
    )


def calculate_take_profit(
    entry_price: float,
    take_profit_percent: float = 4.0,
) -> float:
    """
    Calculate take-profit price for a long position.
    """

    if entry_price <= 0:
        raise ValueError("Entry price must be greater than zero")

    if take_profit_percent <= 0:
        raise ValueError("Take profit percentage must be greater than zero")

    return entry_price * (
        1 + take_profit_percent / 100
    )