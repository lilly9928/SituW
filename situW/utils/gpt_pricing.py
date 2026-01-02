def _standard_prices_per_1m():
    return {
        "gpt-5.2": (1.75, 14.00),
        "gpt-5.1": (1.25, 10.00),
        "gpt-5": (1.25, 10.00),
        "gpt-5-mini": (0.25, 2.00),
        "gpt-5-nano": (0.05, 0.40),
        "gpt-5.2-chat-latest": (1.75, 14.00),
        "gpt-5.1-chat-latest": (1.25, 10.00),
        "gpt-5-chat-latest": (1.25, 10.00),
        "gpt-5.1-codex-max": (1.25, 10.00),
        "gpt-5.1-codex": (1.25, 10.00),
        "gpt-5-codex": (1.25, 10.00),
        "gpt-5.2-pro": (21.00, 168.00),
        "gpt-5-pro": (15.00, 120.00),
        "gpt-4.1": (2.00, 8.00),
        "gpt-4.1-mini": (0.40, 1.60),
        "gpt-4.1-nano": (0.10, 0.40),
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-2024-05-13": (5.00, 15.00),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-realtime": (4.00, 16.00),
        "gpt-realtime-mini": (0.60, 2.40),
        "gpt-4o-realtime-preview": (5.00, 20.00),
        "gpt-4o-mini-realtime-preview": (0.60, 2.40),
        "gpt-audio": (2.50, 10.00),
        "gpt-audio-mini": (0.60, 2.40),
        "gpt-4o-audio-preview": (2.50, 10.00),
        "gpt-4o-mini-audio-preview": (0.15, 0.60),
        "o1": (15.00, 60.00),
        "o1-pro": (150.00, 600.00),
        "o3-pro": (20.00, 80.00),
        "o3": (2.00, 8.00),
        "o3-deep-research": (10.00, 40.00),
        "o4-mini": (1.10, 4.40),
        "o4-mini-deep-research": (2.00, 8.00),
        "o3-mini": (1.10, 4.40),
        "o1-mini": (1.10, 4.40),
        "gpt-5.1-codex-mini": (0.25, 2.00),
        "codex-mini-latest": (1.50, 6.00),
        "gpt-5-search-api": (1.25, 10.00),
        "gpt-4o-mini-search-preview": (0.15, 0.60),
        "gpt-4o-search-preview": (2.50, 10.00),
        "computer-use-preview": (3.00, 12.00)
    }


def get_text_prices_per_1m(model_name, tier="standard"):
    m = (model_name or "").strip().lower()
    t = (tier or "standard").strip().lower()

    if t != "standard":
        t = "standard"

    table = _standard_prices_per_1m()

    if m in table:
        a, b = table[m]
        return {"tier": t, "matched": m, "input_per_1m": float(a), "output_per_1m": float(b)}

    for k, v in table.items():
        if m.startswith(k + "-"):
            a, b = v
            return {"tier": t, "matched": k, "input_per_1m": float(a), "output_per_1m": float(b)}

    return {"tier": t, "matched": None, "input_per_1m": None, "output_per_1m": None}
