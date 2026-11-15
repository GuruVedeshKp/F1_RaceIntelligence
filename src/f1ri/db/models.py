from datetime import date as DateValue
from datetime import datetime as DateTimeValue

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from f1ri.db.base import Base


class Race(Base):
    __tablename__ = "races"

    # Both constraints must be individual elements
    # inside one flat tuple.
    __table_args__ = (
        UniqueConstraint(
            "season",
            "round",
            name="uq_races_season_round",
        ),
        UniqueConstraint(
            "source",
            "source_session_key",
            name="uq_races_source_session",
        ),
    )

    race_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    season: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
    )

    round: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    circuit: Mapped[str] = mapped_column(
        String(120),
        nullable=False,
    )

    date: Mapped[DateValue] = mapped_column(
        Date,
        nullable=False,
    )

    total_laps: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    source: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        server_default=text("'openf1'"),
    )

    source_meeting_key: Mapped[int | None] = mapped_column(Integer)

    source_session_key: Mapped[int | None] = mapped_column(Integer)

    session_start_utc: Mapped[DateTimeValue | None] = mapped_column(
        DateTime(timezone=True)
    )

    session_end_utc: Mapped[DateTimeValue | None] = mapped_column(
        DateTime(timezone=True)
    )


class Lap(Base):
    __tablename__ = "laps"

    __table_args__ = (
        Index("ix_laps_race_lap", "race_id", "lap"),
        Index("ix_laps_driver", "driver"),
    )

    race_id: Mapped[int] = mapped_column(
        ForeignKey("races.race_id", ondelete="CASCADE"),
        primary_key=True,
    )
    driver: Mapped[str] = mapped_column(String(3), primary_key=True)
    lap: Mapped[int] = mapped_column(Integer, primary_key=True)

    lap_time_s: Mapped[float | None] = mapped_column(Float)
    position: Mapped[int | None] = mapped_column(SmallInteger)
    compound: Mapped[str | None] = mapped_column(String(12))
    tyre_age: Mapped[float | None] = mapped_column(Float)
    stint: Mapped[int | None] = mapped_column(SmallInteger)

    pit_in: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    pit_out: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )

    track_status: Mapped[str | None] = mapped_column(String(32))
    is_clean: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )

    driver_number: Mapped[int | None] = mapped_column(Integer)

    lap_start_utc: Mapped[DateTimeValue | None] = mapped_column(
        DateTime(timezone=True)
    )

    lap_end_utc: Mapped[DateTimeValue | None] = mapped_column(
        DateTime(timezone=True)
    )

    position_state_age_s: Mapped[float | None] = mapped_column(Float)


class PitStop(Base):
    __tablename__ = "pit_stops"

    race_id: Mapped[int] = mapped_column(
        ForeignKey("races.race_id", ondelete="CASCADE"),
        primary_key=True,
    )
    driver: Mapped[str] = mapped_column(String(3), primary_key=True)
    lap: Mapped[int] = mapped_column(Integer, primary_key=True)

    pit_loss_s: Mapped[float | None] = mapped_column(Float)
    under_sc: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )

    driver_number: Mapped[int | None] = mapped_column(Integer)

    pit_time_utc: Mapped[DateTimeValue | None] = mapped_column(
        DateTime(timezone=True)
    )

    lane_duration_s: Mapped[float | None] = mapped_column(Float)

    stationary_duration_s: Mapped[float | None] = mapped_column(Float)


class TrackEvent(Base):
    __tablename__ = "track_events"

    event_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    race_id: Mapped[int] = mapped_column(
        ForeignKey("races.race_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    event_type: Mapped[str] = mapped_column("type", String(16), nullable=False)
    start_lap: Mapped[int] = mapped_column(Integer, nullable=False)
    end_lap: Mapped[int | None] = mapped_column(Integer)
    message: Mapped[str | None] = mapped_column(Text)

    event_time_utc: Mapped[DateTimeValue | None] = mapped_column(
        DateTime(timezone=True)
    )

    category: Mapped[str | None] = mapped_column(String(32))

    flag: Mapped[str | None] = mapped_column(String(32))

    scope: Mapped[str | None] = mapped_column(String(32))

    sector: Mapped[int | None] = mapped_column(SmallInteger)

    driver_number: Mapped[int | None] = mapped_column(Integer)


class Weather(Base):
    __tablename__ = "weather"

    race_id: Mapped[int] = mapped_column(
        ForeignKey("races.race_id", ondelete="CASCADE"),
        primary_key=True,
    )
    lap: Mapped[int] = mapped_column(Integer, primary_key=True)

    air_temp: Mapped[float | None] = mapped_column(Float)
    track_temp: Mapped[float | None] = mapped_column(Float)
    rain: Mapped[bool | None] = mapped_column(Boolean)
    weather_age_s: Mapped[float | None] = mapped_column(Float)


class RaceResult(Base):
    __tablename__ = "results"

    race_id: Mapped[int] = mapped_column(
        ForeignKey("races.race_id", ondelete="CASCADE"),
        primary_key=True,
    )
    driver: Mapped[str] = mapped_column(String(3), primary_key=True)

    team: Mapped[str | None] = mapped_column(String(80))
    grid: Mapped[int | None] = mapped_column(SmallInteger)
    finish_pos: Mapped[int | None] = mapped_column(SmallInteger)
    status: Mapped[str | None] = mapped_column(String(40))
    points: Mapped[float | None] = mapped_column(Float)
    driver_number: Mapped[int | None] = mapped_column(Integer)

    number_of_laps: Mapped[int | None] = mapped_column(Integer)

    duration_s: Mapped[float | None] = mapped_column(Float)

    gap_to_leader_raw: Mapped[str | None] = mapped_column(String(40))

    dnf: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )

    dns: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )

    dsq: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )


class LapState(Base):
    __tablename__ = "lap_state"

    __table_args__ = (
        ForeignKeyConstraint(
            ["race_id", "driver", "lap"],
            ["laps.race_id", "laps.driver", "laps.lap"],
            ondelete="CASCADE",
            name="fk_lap_state_lap",
        ),
        Index("ix_lap_state_race_lap", "race_id", "lap"),
    )

    race_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    driver: Mapped[str] = mapped_column(String(3), primary_key=True)
    lap: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Model features
    laps_remaining: Mapped[int | None] = mapped_column(Integer)
    position: Mapped[int | None] = mapped_column(SmallInteger)
    gap_to_leader_s: Mapped[float | None] = mapped_column(Float)
    laps_behind_leader: Mapped[int | None] = mapped_column(SmallInteger)
    gap_to_ahead_s: Mapped[float | None] = mapped_column(Float)
    gap_to_behind_s: Mapped[float | None] = mapped_column(Float)

    compound: Mapped[str | None] = mapped_column(String(12))
    tyre_age: Mapped[float | None] = mapped_column(Float)
    stint: Mapped[int | None] = mapped_column(SmallInteger)
    stops_made: Mapped[int | None] = mapped_column(SmallInteger)

    track_status: Mapped[str | None] = mapped_column(String(32))
    grid_position: Mapped[int | None] = mapped_column(SmallInteger)
    team: Mapped[str | None] = mapped_column(String(80))

    rolling_pace_s: Mapped[float | None] = mapped_column(Float)
    pace_delta_to_leader_s: Mapped[float | None] = mapped_column(Float)

    air_temp: Mapped[float | None] = mapped_column(Float)
    track_temp: Mapped[float | None] = mapped_column(Float)
    rain: Mapped[bool | None] = mapped_column(Boolean)

    circuit: Mapped[str | None] = mapped_column(String(120))
    era: Mapped[str | None] = mapped_column(String(120))

    # Prediction targets
    finish_position: Mapped[int | None] = mapped_column(SmallInteger)
    is_win: Mapped[bool | None] = mapped_column(Boolean)
    is_top3: Mapped[bool | None] = mapped_column(Boolean)
    is_top10: Mapped[bool | None] = mapped_column(Boolean)
    is_dnf: Mapped[bool | None] = mapped_column(Boolean)


class Prediction(Base):
    __tablename__ = "predictions"

    __table_args__ = (
        ForeignKeyConstraint(
            ["race_id", "driver", "lap"],
            ["laps.race_id", "laps.driver", "laps.lap"],
            ondelete="CASCADE",
            name="fk_predictions_lap",
        ),
        CheckConstraint(
            "p_win BETWEEN 0 AND 1",
            name="ck_predictions_p_win_range",
        ),
        CheckConstraint(
            "p_top3 BETWEEN 0 AND 1",
            name="ck_predictions_p_top3_range",
        ),
        CheckConstraint(
            "p_top10 BETWEEN 0 AND 1",
            name="ck_predictions_p_top10_range",
        ),
        CheckConstraint(
            "p_dnf BETWEEN 0 AND 1",
            name="ck_predictions_p_dnf_range",
        ),
        CheckConstraint(
            "p_win <= p_top3 AND p_top3 <= p_top10",
            name="ck_predictions_probability_order",
        ),
    )

    race_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    driver: Mapped[str] = mapped_column(String(3), primary_key=True)
    lap: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_version: Mapped[str] = mapped_column(String(64), primary_key=True)

    p_win: Mapped[float] = mapped_column(Float, nullable=False)
    p_top3: Mapped[float] = mapped_column(Float, nullable=False)
    p_top10: Mapped[float] = mapped_column(Float, nullable=False)
    p_dnf: Mapped[float] = mapped_column(Float, nullable=False)
