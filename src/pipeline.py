"""
Shared data pipeline utilities.
Centralizes fetch → clean workflow used by API, CLI, and backtest.
"""

from typing import Dict, List, Optional

from .data import fetch_multiple_stocks
from .data_cleaning import clean_multiple_stocks
from .logging_setup import get_logger

logger = get_logger(__name__)


def fetch_and_clean(
    tickers: List[str],
    period: str = "3y",
    interval: str = "1d",
    use_cache: bool = True,
    save_permanent: bool = False,
) -> Dict:
    """
    Fetch and clean stock data for one or more tickers.

    Returns:
        Dict mapping ticker symbols to cleaned DataFrames.
    """
    logger.info("Fetching data for %d ticker(s): %s", len(tickers), tickers)
    stock_data = fetch_multiple_stocks(
        tickers, period=period, interval=interval, use_cache=use_cache
    )

    if not stock_data:
        logger.warning("No stock data fetched for tickers: %s", tickers)
        return {}

    logger.info("Cleaning data for %d ticker(s)", len(stock_data))
    cleaned = clean_multiple_stocks(
        stock_data, use_cache=use_cache, save_permanent=save_permanent
    )
    return cleaned


def fetch_and_clean_single(
    ticker: str,
    period: str = "3y",
    interval: str = "1d",
    use_cache: bool = True,
) -> Optional[object]:
    """Fetch and clean data for a single ticker. Returns DataFrame or None."""
    cleaned = fetch_and_clean([ticker], period, interval, use_cache)
    return cleaned.get(ticker)
