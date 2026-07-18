"""PAPER-ONLY Moomoo order broker.

===============================  GUARDRAILS  ===============================
This module can NEVER touch real money, by construction:

 1. The trading environment is the literal constant ``TrdEnv.SIMULATE`` --
    it is not read from config, not a parameter, not overridable. The
    read-only client's ``settings.trd_env`` is deliberately ignored.
 2. On connect it selects an account from ``get_acc_list()`` filtered to
    ``trd_env == SIMULATE`` and refuses to operate if none exists. Every
    order call passes BOTH ``trd_env=SIMULATE`` and that paper ``acc_id``.
 3. ``unlock_trade`` is never imported or called. Moomoo requires an unlock
    for REAL orders, so even a hypothetical bug that constructed a real
    order would be rejected by OpenD itself.
 4. Every order intent and result is appended to a ledger on disk before
    and after submission -- nothing trades silently.

Everything else in the app (moomoo_client.py) remains read-only; this module
is the only file that imports ``place_order``-capable calls, and it is
paper-only. Scaling to real money is a HUMAN decision that would require a
deliberately different module -- do not "upgrade" this one.
=============================================================================
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

try:
    from moomoo import (
        OpenSecTradeContext,
        OrderStatus,
        OrderType,
        RET_OK,
        SecurityFirm,
        TrdEnv,
        TrdMarket,
        TrdSide,
    )
except ImportError:  # pragma: no cover
    from futu import (
        OpenSecTradeContext,
        OrderStatus,
        OrderType,
        RET_OK,
        SecurityFirm,
        TrdEnv,
        TrdMarket,
        TrdSide,
    )

import time

from app.config import settings

PAPER_ENV = TrdEnv.SIMULATE          # the one and only environment. Never change.
LEDGER_DIR = Path(__file__).resolve().parents[2] / "data_store" / "paper"

# Moomoo hard-caps place_order at 15 per 30s. Exceeding it returns REJECTED
# ("high frequency"), and because the entry loop counts SUCCESSES, a rejected
# batch just makes it retry the whole candidate list -- 92 rejects / 56 aborts
# on 2026-07-17. Throttle here at the broker boundary so EVERY order path is
# covered and cannot drift. Margin under the cap absorbs any other client
# (e.g. a second app) drawing on the same per-account budget.
ORDER_RATE_MAX = 13
ORDER_RATE_WINDOW = 30.0


class PaperBrokerError(RuntimeError):
    pass


def _assert_simulate(env) -> None:
    if env != TrdEnv.SIMULATE:
        raise PaperBrokerError("guardrail: refusing non-SIMULATE environment")


class PaperBroker:
    """Order interface bound to the Moomoo SIMULATE (paper) account only."""

    def __init__(self, host: str | None = None, port: int | None = None):
        self.host = host or settings.opend_host
        self.port = port or settings.opend_port
        self._trade: OpenSecTradeContext | None = None
        self.acc_id: int | None = None
        self._order_times: list[float] = []   # sliding window for rate throttle
        LEDGER_DIR.mkdir(parents=True, exist_ok=True)

    def _throttle_order(self) -> None:
        """Block until placing one more order stays under the 15/30s cap."""
        now = time.time()
        self._order_times = [t for t in self._order_times if now - t < ORDER_RATE_WINDOW]
        if len(self._order_times) >= ORDER_RATE_MAX:
            wait = ORDER_RATE_WINDOW - (now - self._order_times[0]) + 0.1
            if wait > 0:
                time.sleep(wait)
            now = time.time()
            self._order_times = [t for t in self._order_times if now - t < ORDER_RATE_WINDOW]
        self._order_times.append(time.time())

    # --- lifecycle -----------------------------------------------------
    def connect(self) -> "PaperBroker":
        if self._trade is None:
            firm = getattr(SecurityFirm, "FUTUINC", None) or next(
                getattr(SecurityFirm, n) for n in dir(SecurityFirm) if n.startswith("FUTU"))
            self._trade = OpenSecTradeContext(
                filter_trdmarket=TrdMarket.US, host=self.host, port=self.port,
                security_firm=firm)
        ret, accs = self._trade.get_acc_list()
        if ret != RET_OK:
            raise PaperBrokerError(f"get_acc_list failed: {accs}")
        paper = accs[accs["trd_env"] == PAPER_ENV]
        if paper.empty:
            raise PaperBrokerError("guardrail: no SIMULATE account found -- refusing to trade")
        self.acc_id = int(paper.iloc[0]["acc_id"])
        return self

    def close(self) -> None:
        if self._trade is not None:
            self._trade.close()
            self._trade = None

    # --- reads (paper account only) --------------------------------------
    def _q(self, ret, data, what):
        if ret != RET_OK:
            raise PaperBrokerError(f"{what} failed: {data}")
        return data

    def positions(self) -> pd.DataFrame:
        _assert_simulate(PAPER_ENV)
        return self._q(*self._trade.position_list_query(
            trd_env=PAPER_ENV, acc_id=self.acc_id), what="position_list_query")

    def orders(self, status_filter_list=None) -> pd.DataFrame:
        _assert_simulate(PAPER_ENV)
        return self._q(*self._trade.order_list_query(
            trd_env=PAPER_ENV, acc_id=self.acc_id,
            status_filter_list=status_filter_list or []), what="order_list_query")

    def account(self) -> pd.DataFrame:
        _assert_simulate(PAPER_ENV)
        return self._q(*self._trade.accinfo_query(
            trd_env=PAPER_ENV, acc_id=self.acc_id), what="accinfo_query")

    # --- orders (paper account only) --------------------------------------
    def place_limit(self, code: str, qty: float, side: str, price: float,
                    note: str = "") -> dict:
        """Limit order on the PAPER account. side: 'BUY' | 'SELL'."""
        _assert_simulate(PAPER_ENV)
        if self.acc_id is None:
            raise PaperBrokerError("not connected")
        trd_side = TrdSide.BUY if side.upper() == "BUY" else TrdSide.SELL
        intent = {
            "ts": dt.datetime.now().isoformat(timespec="seconds"),
            "env": "SIMULATE", "acc_id": self.acc_id, "code": code,
            "side": side.upper(), "qty": qty, "limit": round(price, 2),
            "note": note, "status": "INTENT",
        }
        self._ledger(intent)
        self._throttle_order()          # never exceed Moomoo's 15/30s place-order cap
        ret, data = self._trade.place_order(
            price=round(price, 2), qty=qty, code=code, trd_side=trd_side,
            order_type=OrderType.NORMAL, trd_env=PAPER_ENV, acc_id=self.acc_id)
        if ret != RET_OK:
            intent.update(status="REJECTED", error=str(data))
            self._ledger(intent)
            return intent
        intent.update(status="SUBMITTED",
                      order_id=str(data.iloc[0].get("order_id", "")))
        self._ledger(intent)
        return intent

    def modify_price(self, order_id: str, price: float, qty: float | None = None,
                     note: str = "") -> dict:
        """Amend a RESTING order's price (and optionally qty) in place, instead
        of cancel-then-replace. Preserves the order's queue identity and avoids
        the window where a leg sits cancelled with no working order. qty=None
        keeps the original quantity (the SDK requires a qty, so callers pass the
        current one). Paper account only."""
        _assert_simulate(PAPER_ENV)
        try:
            from moomoo import ModifyOrderOp
        except ImportError:  # pragma: no cover
            from futu import ModifyOrderOp
        if qty is None:
            # look up the working order's qty; the SDK needs an explicit value
            cur = self.orders()
            row = cur[cur["order_id"].astype(str) == str(order_id)] if not cur.empty else cur
            qty = float(row.iloc[0]["qty"]) if not row.empty else 0
        rec = {"ts": dt.datetime.now().isoformat(timespec="seconds"), "env": "SIMULATE",
               "acc_id": self.acc_id, "order_id": order_id, "action": "MODIFY",
               "new_price": round(price, 2), "new_qty": qty, "note": note, "status": "MODIFY_INTENT"}
        self._ledger(rec)
        self._throttle_order()          # modify_order shares the high-frequency cap family
        ret, data = self._trade.modify_order(
            ModifyOrderOp.NORMAL, order_id, qty, round(price, 2),
            trd_env=PAPER_ENV, acc_id=self.acc_id)
        rec = {**rec, "status": "MODIFIED" if ret == RET_OK else "MODIFY_FAILED",
               "error": None if ret == RET_OK else str(data)}
        self._ledger(rec)
        return rec

    def cancel_order(self, order_id: str) -> bool:
        _assert_simulate(PAPER_ENV)
        try:
            from moomoo import ModifyOrderOp
        except ImportError:  # pragma: no cover
            from futu import ModifyOrderOp
        self._throttle_order()          # cancels ride the same modify_order cap
        ret, data = self._trade.modify_order(
            ModifyOrderOp.CANCEL, order_id, 0, 0,
            trd_env=PAPER_ENV, acc_id=self.acc_id)
        self._ledger({"ts": dt.datetime.now().isoformat(timespec="seconds"),
                      "order_id": order_id, "status": "CANCELLED" if ret == RET_OK else "CANCEL_FAILED",
                      "env": "SIMULATE"})
        return ret == RET_OK

    # --- ledger -----------------------------------------------------------
    @staticmethod
    def _ledger(row: dict) -> None:
        path = LEDGER_DIR / "order_ledger.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    b = PaperBroker().connect()
    print(f"paper account: {b.acc_id}")
    acc = b.account()
    cols = [c for c in ("power", "total_assets", "cash", "us_cash") if c in acc.columns]
    print(acc[cols].to_string(index=False) if cols else acc.head(1).to_string())
    pos = b.positions()
    print(f"open positions: {len(pos)}")
    b.close()
