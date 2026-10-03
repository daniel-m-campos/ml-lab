"""Optiver "Trading at the Close" (Kaggle, 2023) as a forestry campaign.

Real order-book columns, 200 stocks, 481 trading days, target is the 60 s ahead move of the
stock's weighted average price in basis points. The competition hides calendar dates, so
``date_id`` is mapped to consecutive calendar days from a base date; monthly folds are then
about thirty trading days wide.

Data: ``data/optiver/optiver-trading-at-the-close/train.csv`` (Kaggle, competition rules or a
public mirror). Run ``python examples/optiver/campaign.py --stocks 20``.
"""
