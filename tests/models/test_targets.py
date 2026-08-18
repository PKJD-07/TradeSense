import numpy as np
import pandas as pd
import pytest

from src.models.targets import create_targets


def make_dataframe():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-08-01",
                periods=10,
                freq="D",
            ),
            "symbol": ["AAPL"] * 10,
            "close": [
                100,
                101,
                102,
                103,
                104,
                105,
                106,
                107,
                108,
                109,
            ],
        }
    )


def test_create_targets_adds_target_columns():
    df = make_dataframe()

    result = create_targets(
        df,
        horizon=1,
        threshold=0.01,
    )

    assert "forward_return" in result.columns
    assert "target" in result.columns


def test_target_return_uses_future_close():
    df = make_dataframe()

    result = create_targets(
        df,
        horizon=1,
        threshold=0.01,
    )

    expected = 101 / 100 - 1

    assert result.loc[0, "forward_return"] == pytest.approx(
        expected
    )


def test_target_return_is_forward_looking():
    df = make_dataframe()

    result = create_targets(
        df,
        horizon=2,
        threshold=0.01,
    )

    expected = 102 / 100 - 1

    assert result.loc[0, "forward_return"] == pytest.approx(
        expected
    )


def test_target_direction_is_three_class():
    df = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 6,
            "close": [
                100,
                110,
                100,
                100,
                90,
                100,
            ],
        }
    )

    result = create_targets(
        df,
        horizon=1,
        threshold=0.05,
    )

    assert result.loc[0, "target"] == 1
    assert result.loc[1, "target"] == -1
    assert result.loc[2, "target"] == 0


def test_last_row_has_no_future_target():
    df = make_dataframe()

    result = create_targets(
        df,
        horizon=1,
        threshold=0.01,
    )

    assert pd.isna(result.loc[9, "forward_return"])
    assert pd.isna(result.loc[9, "target"])


def test_targets_are_generated_per_symbol():
    df = pd.DataFrame(
        {
            "symbol": [
                "AAPL",
                "AAPL",
                "AAPL",
                "MSFT",
                "MSFT",
                "MSFT",
            ],
            "close": [
                100,
                110,
                120,
                200,
                180,
                160,
            ],
        }
    )

    result = create_targets(
        df,
        horizon=1,
        threshold=0.05,
    )

    assert result.loc[0, "target"] == 1
    assert result.loc[1, "target"] == 1

    assert result.loc[3, "target"] == -1
    assert result.loc[4, "target"] == -1

    assert pd.isna(result.loc[2, "target"])
    assert pd.isna(result.loc[5, "target"])


def test_invalid_horizon_raises_error():
    df = make_dataframe()

    with pytest.raises(ValueError, match="horizon"):
        create_targets(df, horizon=0)


def test_negative_threshold_raises_error():
    df = make_dataframe()

    with pytest.raises(ValueError, match="threshold"):
        create_targets(df, threshold=-0.01)


def test_missing_required_columns_raises_error():
    df = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "open": [100],
        }
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        create_targets(df)


def test_target_values_are_three_class():
    df = make_dataframe()

    result = create_targets(
        df,
        horizon=1,
        threshold=0.01,
    )

    valid_targets = result["target"].dropna().unique()

    assert set(valid_targets).issubset({-1, 0, 1})