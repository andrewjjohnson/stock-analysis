"""Offline demo on SYNTHETIC data: no API key, no network.

Runs the same pipeline as run.py (EMA single config, sweep and split; opening-range
reversal baseline and threshold sweep with split, both with barriers; the auction reclaim
comparison with split and barriers), then sauce.py's VWAP + Sauce simulation (the brief's
defaults, and the four --compare variants with every optional module on), on a seeded random
walk laid on the real XNYS calendar, labeled ticker SYNTHETIC. It checks the
workflow end to end; its numbers say nothing about any real market.

  uv run python demo.py
"""

import features
import run
import sauce
import synthetic

START, END, SPLIT = "2024-01-02", "2024-12-31", "2024-07-01"
SOURCE = "SYNTHETIC random walk (demo only, not market data)"
RUNS = {
    "single": [],
    "sweep": ["--fast", "5", "9", "12", "--slow", "20", "21", "30"],
    "split": ["--fast", "5", "9", "12", "--slow", "20", "21", "30", "--split-date", SPLIT],
    "orr": ["--strategy", "opening_range_reversal", "--barriers"],
    "orr_split": ["--strategy", "opening_range_reversal", "--barriers", "--threshold", "0.20", "0.25", "0.30",
                  "--split-date", SPLIT],
    "auction_reclaim": ["--strategy", "auction_reclaim", "--barriers", "--compare", "--split-date", SPLIT],
}
SAUCE_RUNS = {
    "sauce": [],
    "sauce_all": ["--compare", "--continuation", "--vwap-to-vwap", "--reentry", "--fade-entry", "--trendline-confirm",
                  "--random-baseline", "20"],
}


def main():
    timings = {}
    with run.timed(timings, "data"):
        sessions = features.trading_sessions(START, END, warmup_sessions=120)
        minutes = synthetic.random_walk_minutes(sessions, seed=7)
    for name, extra in RUNS.items():
        print(f"\n{'=' * 30} DEMO: {name} · SYNTHETIC DATA, NOT MARKET DATA {'=' * 30}")
        args = run.parse_args(["--ticker", "SYNTHETIC", "--start", START, "--end", END,
                               "--out", f"output/demo/{name}", *extra])
        run.execute(args, minutes, sessions, SOURCE, dict(timings))
    for name, extra in SAUCE_RUNS.items():
        print(f"\n{'=' * 30} DEMO: {name} · SYNTHETIC DATA, NOT MARKET DATA {'=' * 30}")
        args = sauce.parse_args(["--ticker", "SYNTHETIC", "--start", START, "--end", END, "--charts", "3",
                                 "--out", f"output/demo/{name}", *extra])
        sauce.execute(args, minutes, sessions, SOURCE, dict(timings))


if __name__ == "__main__":
    main()
