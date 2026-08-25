def generate_signal(analysis: dict) -> dict:
    score = 0
    reasons = []

    # Price vs 20-period SMA
    if analysis["latest_price"] > analysis["sma_20"]:
        score += 1
        reasons.append("Price is above the 20-period SMA")
    else:
        score -= 1
        reasons.append("Price is below the 20-period SMA")

    # Price vs 20-period EMA
    if analysis["latest_price"] > analysis["ema_20"]:
        score += 1
        reasons.append("Price is above the 20-period EMA")
    else:
        score -= 1
        reasons.append("Price is below the 20-period EMA")

    # Price vs 50-period SMA
    if analysis["latest_price"] > analysis["sma_50"]:
        score += 1
        reasons.append("Price is above the 50-period SMA")
    else:
        score -= 1
        reasons.append("Price is below the 50-period SMA")

    # RSI momentum
    if analysis["rsi_14"] > 50:
        score += 1
        reasons.append("RSI is above 50")
    else:
        score -= 1
        reasons.append("RSI is below 50")

    # MACD momentum
    if analysis["macd"] > analysis["macd_signal"]:
        score += 1
        reasons.append("MACD is above the signal line")
    else:
        score -= 1
        reasons.append("MACD is below the signal line")

    # ADX trend strength
    if analysis["adx_14"] >= 20:
        trend_confirmed = True
        reasons.append("ADX confirms a strong enough trend")
    else:
        trend_confirmed = False
        reasons.append("ADX indicates a weak trend")

    # Final decision
    # Score range: -5 to +5
    if score >= 3:
        decision = "BUY"
    elif score <= -3:
        decision = "SELL"
    else:
        decision = "HOLD"

    return {
        "decision": decision,
        "score": score,
        "reasons": reasons,
    }
