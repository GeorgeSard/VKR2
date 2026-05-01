"""Definitions of feature sets used across experiments.

Three nested sets, each strictly a superset of the previous one. Switching
between them (with model and seed held fixed) is one of the demo axes for
the MLflow comparison: every set should produce a visibly different metric.

Design intent for the demo:
- BASIC          — only schedule-known fields. No inbound delay, no
                   congestion. Deliberately weak so the next sets have
                   room to demonstrate gains.
- EXTENDED       — adds the two strongest non-weather signals: cascading
                   delay from the previous flight leg of the same aircraft
                   (`inbound_delay_minutes`) and airport congestion indices.
- WITH_WEATHER   — adds origin/destination weather forecast features.

LEAKAGE_COLUMNS lists post-event columns that must NEVER appear in any
feature set. Mirrors params.yaml `data.leakage_columns` and the dataset
documentation in DATA_DICTIONARY.md.
"""

from __future__ import annotations

from dataclasses import dataclass

# Post-event columns: known only after the flight has happened.
# Including any of these in features = target leakage = inflated metrics.
LEAKAGE_COLUMNS: frozenset[str] = frozenset(
    {
        "gt_carrier_delay_minutes",
        "gt_weather_delay_minutes",
        "gt_airport_congestion_delay_minutes",
        "gt_reactionary_delay_minutes",
        "gt_security_delay_minutes",
        "dep_delay_minutes",
        "arr_delay_minutes",
        "is_departure_delayed_15m",
        "is_arrival_delayed_15m",
        "actual_departure_local",
        "actual_arrival_local",
        "cancellation_flag",
        "cancellation_reason",
        "diversion_flag",
        "probable_delay_cause",
    }
)


@dataclass(frozen=True)
class FeatureSet:
    name: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]

    @property
    def all_columns(self) -> tuple[str, ...]:
        return self.numeric + self.categorical


_BASIC_NUMERIC = (
    "month",
    "day_of_week",
    "scheduled_dep_hour",
    "scheduled_dep_minute",
    "is_weekend",
    "is_holiday_window",
    "quarter",
    "distance_km",
    "planned_block_minutes",
    "airline_fleet_avg_age",
    "origin_hub_tier",
    "destination_hub_tier",
)

_BASIC_CATEGORICAL = (
    "airline_code",
    "aircraft_family",
    "origin_iata",
    "destination_iata",
    "route_group",
)

_EXTENDED_NUMERIC = _BASIC_NUMERIC + (
    "inbound_delay_minutes",
    "origin_congestion_index",
    "destination_congestion_index",
)

_WEATHER_NUMERIC = _EXTENDED_NUMERIC + (
    "origin_temperature_c",
    "origin_precip_mm",
    "origin_visibility_km",
    "origin_wind_mps",
    "destination_temperature_c",
    "destination_precip_mm",
    "destination_visibility_km",
    "destination_wind_mps",
)

_WEATHER_CATEGORICAL = _BASIC_CATEGORICAL + (
    "origin_weather_severity",
    "destination_weather_severity",
)


FEATURE_SETS: dict[str, FeatureSet] = {
    "basic": FeatureSet("basic", _BASIC_NUMERIC, _BASIC_CATEGORICAL),
    "extended": FeatureSet("extended", _EXTENDED_NUMERIC, _BASIC_CATEGORICAL),
    "with_weather": FeatureSet("with_weather", _WEATHER_NUMERIC, _WEATHER_CATEGORICAL),
}


def get_feature_set(name: str) -> FeatureSet:
    if name not in FEATURE_SETS:
        raise KeyError(
            f"Unknown feature set '{name}'. Available: {sorted(FEATURE_SETS)}. "
            "Update params.yaml → features.active_set."
        )
    return FEATURE_SETS[name]


def assert_no_leakage(columns: list[str]) -> None:
    bad = set(columns) & LEAKAGE_COLUMNS
    if bad:
        raise ValueError(
            f"Feature columns include leakage targets: {sorted(bad)}. "
            "Remove them from feature_sets.py."
        )
