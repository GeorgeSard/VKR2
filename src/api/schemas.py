"""Pydantic schemas for FastAPI input/output.

Single input schema `FlightFeatures` covers both heads — they share the
same `with_weather` feature set (30 columns). Field names match the raw
dataset columns expected by the trained Pipeline's preprocessor.

We use `Field(...)` with examples so the auto-generated /docs page is
useful out of the box: copy the example body, send POST, get prediction.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FlightFeatures(BaseModel):
    """Inputs required by both heads (binary + cause)."""

    # Schedule fields
    month: int = Field(..., ge=1, le=12, examples=[7])
    day_of_week: int = Field(..., ge=0, le=6, examples=[3], description="Monday=0 ... Sunday=6")
    scheduled_dep_hour: int = Field(..., ge=0, le=23, examples=[14])
    scheduled_dep_minute: int = Field(..., ge=0, le=59, examples=[30])
    is_weekend: int = Field(..., ge=0, le=1, examples=[0])
    is_holiday_window: int = Field(..., ge=0, le=1, examples=[0])
    quarter: int = Field(..., ge=1, le=4, examples=[3])

    # Route fields
    distance_km: float = Field(..., gt=0, examples=[2300.0])
    planned_block_minutes: float = Field(..., gt=0, examples=[195.0])

    # Aircraft / airline
    airline_fleet_avg_age: float = Field(..., ge=0, examples=[12.5])
    origin_hub_tier: int = Field(..., examples=[1], description="1=major hub, 2=regional, 3=secondary")
    destination_hub_tier: int = Field(..., examples=[2])

    # Operational state
    inbound_delay_minutes: float = Field(..., examples=[0.0])
    origin_congestion_index: float = Field(..., ge=0, examples=[0.45])
    destination_congestion_index: float = Field(..., ge=0, examples=[0.32])

    # Origin weather
    origin_temperature_c: float = Field(..., examples=[18.0])
    origin_precip_mm: float = Field(..., ge=0, examples=[0.0])
    origin_visibility_km: float = Field(..., ge=0, examples=[10.0])
    origin_wind_mps: float = Field(..., ge=0, examples=[3.5])

    # Destination weather
    destination_temperature_c: float = Field(..., examples=[22.0])
    destination_precip_mm: float = Field(..., ge=0, examples=[0.0])
    destination_visibility_km: float = Field(..., ge=0, examples=[10.0])
    destination_wind_mps: float = Field(..., ge=0, examples=[2.8])

    # Categorical
    airline_code: str = Field(..., examples=["SU"])
    aircraft_family: str = Field(..., examples=["A320"])
    origin_iata: str = Field(..., examples=["SVO"])
    destination_iata: str = Field(..., examples=["LED"])
    route_group: str = Field(..., examples=["domestic_trunk"])
    origin_weather_severity: str = Field(..., examples=["calm"])
    destination_weather_severity: str = Field(..., examples=["calm"])


class DelayPrediction(BaseModel):
    """Output of POST /predict/delay (binary head)."""

    is_delayed: bool = Field(..., description="True if the model predicts >15 min delay")
    delay_probability: float = Field(..., ge=0, le=1, description="P(delay > 15 min)")
    model_name: str = Field(..., examples=["flight-delay-binary"])
    model_version: str = Field(..., examples=["1"])


class CausePrediction(BaseModel):
    """Output of POST /predict/cause (multi-class head)."""

    predicted_cause: str = Field(..., examples=["weather"])
    class_probabilities: dict[str, float] = Field(
        ...,
        description="P(cause) for each class — sums to ~1",
        examples=[
            {
                "airport_congestion": 0.04,
                "cancelled": 0.01,
                "carrier_operational": 0.07,
                "none": 0.62,
                "reactionary": 0.05,
                "security": 0.005,
                "weather": 0.21,
            }
        ],
    )
    model_name: str = Field(..., examples=["flight-delay-cause"])
    model_version: str = Field(..., examples=["1"])


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    binary_loaded: bool
    cause_loaded: bool


class ModelInfo(BaseModel):
    name: str
    version: str
    run_id: str
    git_commit: str | None = None
    dvc_data_hash: str | None = None
    feature_set: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)


class ModelInfoResponse(BaseModel):
    binary: ModelInfo
    cause: ModelInfo
