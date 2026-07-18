"""Self-check for the place_order rate throttle (paper_broker._throttle_order).

Shrinks the window so it runs in ~2s. Run:
  PYTHONPATH=. .venv/Scripts/python.exe app/brokers/test_order_throttle.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import app.brokers.paper_broker as pb  # noqa: E402


def demo() -> None:
    pb.ORDER_RATE_MAX = 5
    pb.ORDER_RATE_WINDOW = 1.0

    b = pb.PaperBroker()          # __init__ only; no OpenD connection needed
    n = 12
    ts = []                       # record each release (_order_times prunes itself)
    for _ in range(n):
        b._throttle_order()
        ts.append(time.time())
    assert len(ts) == n

    # invariant: no sliding window of WINDOW seconds holds more than MAX releases
    for t in ts:
        in_win = [x for x in ts if t <= x < t + pb.ORDER_RATE_WINDOW]
        assert len(in_win) <= pb.ORDER_RATE_MAX, \
            f"{len(in_win)} orders in a {pb.ORDER_RATE_WINDOW}s window > cap {pb.ORDER_RATE_MAX}"

    # and it actually throttled (12 orders at 5/s can't finish in under ~2s)
    span = ts[-1] - ts[0]
    assert span >= (n / pb.ORDER_RATE_MAX - 1) * pb.ORDER_RATE_WINDOW, \
        f"span {span:.2f}s too short -- throttle didn't engage"

    print(f"order throttle: {n} calls held to <= {pb.ORDER_RATE_MAX} per "
          f"{pb.ORDER_RATE_WINDOW}s over {span:.2f}s -- cap respected")


if __name__ == "__main__":
    demo()
