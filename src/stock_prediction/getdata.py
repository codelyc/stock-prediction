"""Data acquisition helpers for the stock_prediction package (Akshare only)."""
from __future__ import annotations

import argparse
import datetime
import random
import re
from pathlib import Path
from typing import Iterable, Sequence

try:
    from .init import (
        TQDM_NCOLS,
        NoneDataFrame,
        daily_path,
        pd,
        stock_data_queue,
        stock_list_queue,
        threading,
        time,
        tqdm,
    )
except ImportError:  # pragma: no cover
    import sys

    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent.parent
    src_dir = root_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from stock_prediction.init import (
        TQDM_NCOLS,
        NoneDataFrame,
        daily_path,
        pd,
        stock_data_queue,
        stock_list_queue,
        threading,
        time,
        tqdm,
    )

try:
    import akshare as ak
except ImportError:
    ak = None

class DataConfig:
    """Runtime switches controlling fetch behaviour."""

    def __init__(self) -> None:
        self.adjust = "hfq"
        self.code = ""


config = DataConfig()


def set_adjust(adjust: str) -> None:
    """Update the adjustment flag used for downstream fetch operations."""

    config.adjust = adjust


def _normalize_stock_code(code: str) -> str:
    """Canonicalise user-provided symbols to 6-digit A-share codes."""

    raw = str(code).strip()
    if not raw:
        return raw
    raw = raw.split(".", 1)[0]
    digits = re.search(r"\d+", raw)
    if digits is not None:
        return digits.group(0).zfill(6)
    return raw.upper()


def get_stock_list() -> Sequence[str]:
    """Return stock codes from akshare."""

    if ak is None:
        raise ImportError("akshare not installed")
    stock_frame = ak.stock_zh_a_spot_em()
    code_col = "代码" if "代码" in stock_frame.columns else "code" if "code" in stock_frame.columns else stock_frame.columns[0]
    stock_list = (
        stock_frame[code_col]
        .astype(str)
        .map(_normalize_stock_code)
        .dropna()
        .loc[lambda s: s != ""]
        .tolist()
    )
    stock_list_queue.put(stock_list)
    return stock_list


def _iterable_from_code(ts_code: str | Sequence[str]) -> Iterable[str]:
    if isinstance(ts_code, str):
        if ts_code:
            return [ts_code]
        return []
    return ts_code


def get_stock_data(ts_code: Sequence[str] | str = "", save: bool = True, start_code: str = "", save_path: Path | str = "", datediff: int = -1):
    """Download historical bar data for symbols via akshare."""

    if isinstance(save_path, str):
        save_path = Path(save_path)

    if ak is None:
        raise ImportError("akshare not installed")

    stock_list = list(_iterable_from_code(ts_code)) or get_stock_list()
    stock_list = [_normalize_stock_code(code) for code in stock_list if str(code).strip()]
    if start_code:
        normalized_start = _normalize_stock_code(start_code)
        stock_list = stock_list[stock_list.index(normalized_start):]
    pbar = tqdm(total=len(stock_list), leave=False, ncols=TQDM_NCOLS) if save else None
    lock = threading.Lock()

    with lock:
        end_date = (datetime.datetime.now() + datetime.timedelta(days=datediff)).strftime("%Y%m%d")
        for code in stock_list:
            try:
                akshare_code = _normalize_stock_code(code)
                df = ak.stock_zh_a_hist(symbol=akshare_code, period="daily", end_date=end_date, adjust=config.adjust)

                if df.empty:
                    continue

                df.columns = [
                    "trade_date",
                    "ts_code",
                    "open",
                    "close",
                    "high",
                    "low",
                    "vol",
                    "amount",
                    "amplitude",
                    "pct_change",
                    "change",
                    "exchange_rate",
                ]
                columns = list(df.columns)
                columns[0], columns[1] = columns[1], columns[0]
                df = df[columns]
                df["ts_code"] = df["ts_code"].astype(str).map(_normalize_stock_code)
                df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y%m%d")
                df.sort_values(by=["trade_date"], ascending=False, inplace=True)
                df = df.reindex(columns=[
                    "ts_code",
                    "trade_date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "change",
                    "pct_change",
                    "vol",
                    "amount",
                    "amplitude",
                    "exchange_rate",
                ])
            except Exception as exc:  # pragma: no cover
                message = f"{_normalize_stock_code(code)} {exc}"
                if save and pbar is not None:
                    tqdm.write(message)
                    pbar.update(1)
                else:
                    print(message)
                if getattr(exc, "args", []) and isinstance(exc.args[0], Exception):
                    inner = exc.args[0]
                    text = str(inner)
                    if "Connection aborted" in text or "Remote end closed connection" in text:
                        break
                continue

            time.sleep(random.uniform(0.1, 0.9))
            if save:
                save_path.mkdir(parents=True, exist_ok=True)
                df.to_csv(save_path / f"{_normalize_stock_code(code)}.csv", index=False)
                if pbar is not None:
                    pbar.update(1)
            else:
                stock_data_queue.put(df if not df.empty else NoneDataFrame)
                return df if not df.empty else None

    if pbar is not None:
        pbar.close()
    return None


def main() -> None:
    """Command-line entry point for fetching quote data."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--code", default="", type=str, help="single stock code or ticker")
    parser.add_argument("--adjust", default="hfq", type=str, help="adjustment: none, qfq, or hfq")
    args = parser.parse_args()

    config.adjust = args.adjust
    config.code = args.code

    if args.code:
        get_stock_data(args.code, save=True, save_path=daily_path)
    else:
        get_stock_data("", save=True, save_path=daily_path, datediff=-1)


if __name__ == "__main__":
    main()
