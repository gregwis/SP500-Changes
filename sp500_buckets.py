#!/usr/bin/env python3
"""
Rozkład rocznych zmian kursów spółek z indeksu S&P 500.

1. Pobiera aktualny skład S&P 500 (Wikipedia).
2. Pobiera notowania każdej spółki (Yahoo Finance, yfinance).
3. Liczy zmianę kursu w ciągu ostatniego roku.
4. Dzieli spółki na buckety i wypisuje liczbę oraz % spółek w każdym z nich
   względem liczby spółek w całym indeksie.

Użycie:
    pip install yfinance pandas lxml requests
    python sp500_buckets.py [--total-return] [--csv wyniki.csv]
"""

import argparse
import io
import sys
from datetime import timedelta

import pandas as pd
import requests
import yfinance as yf

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# (etykieta, dolna granica włącznie, górna granica wyłącznie) w %
BUCKETS = [
    ("< -20%", float("-inf"), -20),
    ("-20% .. -10%", -20, -10),
    ("-10% .. 0%", -10, 0),
    ("0% .. 10%", 0, 10),
    ("10% .. 20%", 10, 20),
    ("> 20%", 20, float("inf")),
]


def get_sp500_constituents() -> pd.DataFrame:
    """Zwraca DataFrame z kolumnami: Symbol, Security, GICS Sector."""
    headers = {"User-Agent": "Mozilla/5.0 (sp500-buckets script)"}
    resp = requests.get(WIKI_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    table = pd.read_html(io.StringIO(resp.text), attrs={"id": "constituents"})[0]
    df = table[["Symbol", "Security", "GICS Sector"]].copy()
    # Yahoo używa '-' zamiast '.' (np. BRK.B -> BRK-B)
    df["Ticker"] = df["Symbol"].str.replace(".", "-", regex=False)
    return df


def download_prices(tickers: list[str], total_return: bool) -> pd.DataFrame:
    """Zwraca DataFrame cen zamknięcia (wiersze = daty, kolumny = tickery)."""
    data = yf.download(
        tickers,
        period="13mo",  # zapas ponad rok, żeby na pewno złapać datę sprzed roku
        interval="1d",
        auto_adjust=total_return,  # True = ceny skorygowane o dywidendy
        group_by="column",
        threads=True,
        progress=False,
    )
    prices = data["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    return prices.sort_index()


def one_year_change(prices: pd.DataFrame) -> pd.Series:
    """Zmiana % między ostatnim zamknięciem a zamknięciem sprzed roku."""
    changes = {}
    for ticker in prices.columns:
        s = prices[ticker].dropna()
        if s.empty:
            changes[ticker] = float("nan")
            continue
        last_date = s.index[-1]
        target = last_date - pd.DateOffset(years=1)
        past = s[s.index <= target]
        # Brak notowań sprzed roku (np. świeży debiut) albo dziura w danych
        if past.empty or (target - past.index[-1]) > timedelta(days=7):
            changes[ticker] = float("nan")
            continue
        changes[ticker] = (s.iloc[-1] / past.iloc[-1] - 1) * 100
    return pd.Series(changes, name="Change1Y")


def assign_bucket(change: float) -> str | None:
    if pd.isna(change):
        return None
    for label, lo, hi in BUCKETS:
        if lo <= change < hi:
            return label
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--total-return", action="store_true",
                        help="licz zmianę z cen skorygowanych o dywidendy "
                             "(domyślnie: zmiana samego kursu)")
    parser.add_argument("--csv", metavar="PLIK",
                        help="zapisz szczegółowe wyniki per spółka do CSV")
    args = parser.parse_args()

    print("Pobieram skład S&P 500...", file=sys.stderr)
    const = get_sp500_constituents()
    tickers = const["Ticker"].tolist()
    print(f"  {len(tickers)} spółek w indeksie", file=sys.stderr)

    print("Pobieram notowania...", file=sys.stderr)
    prices = download_prices(tickers, args.total_return)

    changes = one_year_change(prices)
    const["Change1Y"] = const["Ticker"].map(changes)
    const["Bucket"] = const["Change1Y"].map(assign_bucket)

    total = len(const)
    print()
    print(f"Okres: {prices.index[0].date()} .. {prices.index[-1].date()} "
          f"(zmiana {'total return' if args.total_return else 'kursu'} r/r)")
    print(f"Liczba spółek w indeksie: {total}")
    print()
    print(f"{'Bucket':<15}{'Liczba':>8}{'% indeksu':>12}")
    print("-" * 35)
    for label, _, _ in BUCKETS:
        n = int((const["Bucket"] == label).sum())
        print(f"{label:<15}{n:>8}{n / total * 100:>11.1f}%")

    missing = const[const["Bucket"].isna()]
    if not missing.empty:
        n = len(missing)
        print(f"{'brak danych':<15}{n:>8}{n / total * 100:>11.1f}%")
    print("-" * 35)
    print(f"{'Razem':<15}{total:>8}{100:>11.1f}%")

    valid = const["Change1Y"].dropna()
    print()
    print(f"Mediana zmiany: {valid.median():.1f}%, średnia: {valid.mean():.1f}%")
    if not missing.empty:
        print("Brak pełnego roku notowań: " + ", ".join(missing["Symbol"]))

    if args.csv:
        const.sort_values("Change1Y", ascending=False).to_csv(args.csv, index=False)
        print(f"Szczegóły zapisane do {args.csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
