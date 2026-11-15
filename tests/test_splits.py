import pandas as pd
import pytest

from f1ri.eval.splits import split_by_season


def make_sample_data() -> pd.DataFrame:
    """Create synthetic races spanning the required seasons."""

    races = [
        (101, 2018, 2),
        (202, 2022, 3),
        (303, 2023, 2),
        (404, 2024, 4),
        (505, 2025, 2),
    ]

    records = []

    for race_id, season, lap_count in races:
        for lap in range(1, lap_count + 1):
            records.append({
                "race_id": race_id,
                "season": season,
                "driver": "VER",
                "lap": lap,
            })

    return pd.DataFrame(records)


def test_season_split_assigns_correct_seasons():
    frame = make_sample_data()

    splits = split_by_season(frame)

    assert set(splits.train["season"]) == {2018, 2022}
    assert set(splits.validation["season"]) == {2023}
    assert set(splits.test["season"]) == {2024, 2025}


def test_all_rows_are_preserved():
    frame = make_sample_data()

    splits = split_by_season(frame)

    total = (
        len(splits.train)
        + len(splits.validation)
        + len(splits.test)
    )

    assert total == len(frame)


def test_races_never_cross_partitions():
    frame = make_sample_data()

    splits = split_by_season(frame)

    train_ids = set(splits.train["race_id"])
    validation_ids = set(splits.validation["race_id"])
    test_ids = set(splits.test["race_id"])

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)


def test_2024_only_data_is_allowed():
    """Current Bahrain coverage belongs to the test partition."""

    frame = pd.DataFrame({
        "race_id": [1, 1, 1],
        "season": [2024, 2024, 2024],
        "lap": [1, 2, 3],
    })

    splits = split_by_season(frame)

    assert splits.train.empty
    assert splits.validation.empty
    assert len(splits.test) == 3


def test_require_nonempty_rejects_missing_partitions():
    frame = pd.DataFrame({
        "race_id": [1],
        "season": [2024],
    })

    with pytest.raises(ValueError, match="Empty partitions"):
        split_by_season(frame, require_nonempty=True)


def test_missing_season_is_rejected():
    frame = pd.DataFrame({
        "race_id": [1, 2],
        "season": [2024, None],
    })

    with pytest.raises(ValueError, match="missing values"):
        split_by_season(frame)


def test_missing_race_id_is_rejected():
    frame = pd.DataFrame({
        "race_id": [1, None],
        "season": [2024, 2024],
    })

    with pytest.raises(ValueError, match="Race ID"):
        split_by_season(frame)


def test_race_cannot_belong_to_multiple_seasons():
    frame = pd.DataFrame({
        "race_id": [1, 1],
        "season": [2023, 2024],
    })

    with pytest.raises(ValueError, match="multiple seasons"):
        split_by_season(frame)


def test_unsupported_seasons_are_rejected():
    frame = pd.DataFrame({
        "race_id": [1],
        "season": [2017],
    })

    with pytest.raises(ValueError, match="do not belong"):
        split_by_season(frame)


def test_invalid_boundaries_are_rejected():
    frame = make_sample_data()

    with pytest.raises(ValueError, match="Invalid split boundaries"):
        split_by_season(
            frame,
            train_start=2018,
            train_end=2023,
            validation_season=2023,
            test_start=2024,
        )