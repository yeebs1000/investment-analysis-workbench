"""Read-only broker interface. Implemented by MoomooClient today; an IBKR
adapter can satisfy the same surface later without touching analytics/API.

By design there is no order-placing method here — the whole app is read-only.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BrokerClient(ABC):
    @abstractmethod
    def get_account_info(self) -> pd.DataFrame: ...

    @abstractmethod
    def get_positions(self) -> pd.DataFrame: ...

    @abstractmethod
    def get_snapshot(self, symbols: list[str]) -> pd.DataFrame: ...

    @abstractmethod
    def get_history_kline(self, symbol: str, **kwargs) -> pd.DataFrame: ...

    @abstractmethod
    def get_watchlist_groups(self) -> pd.DataFrame: ...

    @abstractmethod
    def get_watchlist(self, group_name: str) -> pd.DataFrame: ...
