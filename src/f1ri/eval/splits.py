from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SeasonSplits:
    """Container for chronological training, validation and test datasets."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_by_season(
    frame: pd.DataFrame,
    *,
    season_col: str = "season",
    race_col: str = "race_id",
    train_start: int = 2018,
    train_end: int = 2022,
    validation_season: int = 2023,
    test_start: int = 2024,
    require_nonempty: bool = False,
) -> SeasonSplits:
    """
    Split race data chronologically by season.

    Default policy:
        Train:      2018–2022
        Validation: 2023
        Test:       2024 onward

    All rows belonging to a race remain in the same partition.

    Empty partitions are allowed by default because historical coverage
    may be incomplete while the data pipeline is being developed.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")

    if frame.empty:
        raise ValueError("Cannot split an empty DataFrame.")

    required = {season_col, race_col}
    missing = required - set(frame.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if frame[season_col].isna().any():
        raise ValueError("Season column contains missing values.")

    if frame[race_col].isna().any():
        raise ValueError("Race ID column contains missing values.")

    if not train_start <= train_end < validation_season < test_start:
        raise ValueError(
            "Invalid split boundaries. Expected "
            "train_start <= train_end < validation_season < test_start."
        )

    # Validate that seasons contain integer-like values.
    numeric_seasons = pd.to_numeric(
        frame[season_col],
        errors="coerce",
    )

    if numeric_seasons.isna().any():
        raise ValueError("Season column contains non-numeric values.")

    if (numeric_seasons % 1 != 0).any():
        raise ValueError("Season values must be whole numbers.")

    seasons = numeric_seasons.astype("int64")

    # A race must belong to exactly one season.
    race_seasons = pd.DataFrame({
        "race_id": frame[race_col].to_numpy(),
        "season": seasons.to_numpy(),
    })

    season_counts = race_seasons.groupby("race_id")["season"].nunique()

    if (season_counts > 1).any():
        raise ValueError("A race_id is associated with multiple seasons.")

    # Build chronological masks.
    train_mask = seasons.between(train_start, train_end)

    validation_mask = seasons.eq(validation_season)

    test_mask = seasons.ge(test_start)

    assigned_mask = train_mask | validation_mask | test_mask

    if not assigned_mask.all():
        unsupported = sorted(
            seasons.loc[~assigned_mask].unique().tolist()
        )
        raise ValueError(
            f"Some seasons do not belong to a configured split: {unsupported}"
        )

    splits = SeasonSplits(
        train=frame.loc[train_mask].copy(),
        validation=frame.loc[validation_mask].copy(),
        test=frame.loc[test_mask].copy(),
    )

    # Every source row must appear in exactly one partition.
    total_rows = (
        len(splits.train)
        + len(splits.validation)
        + len(splits.test)
    )

    if total_rows != len(frame):
        raise RuntimeError("Split operation lost or duplicated rows.")

    # Confirm that no race crosses a partition boundary.
    train_races = set(splits.train[race_col])
    validation_races = set(splits.validation[race_col])
    test_races = set(splits.test[race_col])

    if train_races & validation_races:
        raise RuntimeError("Race leakage between train and validation.")

    if train_races & test_races:
        raise RuntimeError("Race leakage between train and test.")

    if validation_races & test_races:
        raise RuntimeError("Race leakage between validation and test.")

    if require_nonempty:
        empty_partitions = [
            name
            for name, part in [
                ("train", splits.train),
                ("validation", splits.validation),
                ("test", splits.test),
            ]
            if part.empty
        ]

        if empty_partitions:
            raise ValueError(
                f"Empty partitions: {empty_partitions}"
            )

    return splits