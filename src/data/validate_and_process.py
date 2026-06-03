"""
Hotel cancellation ETL — read raw CSV from S3, validate, transform, write Parquet.

Run:  uv run python src/data/validate_and_process.py
Exits non-zero if validation fails.
"""

from __future__ import annotations

import logging
import sys

import pandas as pd
import great_expectations as gx
from great_expectations import expectations as gxe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("etl")

ACCOUNT = "803031032486"
RAW_S3 = f"s3://hotel-cancellation-raw-{ACCOUNT}/v1/hotels.csv"
PROCESSED_S3 = f"s3://hotel-cancellation-processed-{ACCOUNT}/v1/hotels.parquet"

EXPECTED_COLUMNS = [
    "hotel", "is_canceled", "lead_time", "arrival_date_year",
    "arrival_date_month", "arrival_date_week_number",
    "arrival_date_day_of_month", "stays_in_weekend_nights",
    "stays_in_week_nights", "adults", "children", "babies",
    "meal", "country", "market_segment", "distribution_channel",
    "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "reserved_room_type",
    "assigned_room_type", "booking_changes", "deposit_type",
    "agent", "company", "days_in_waiting_list", "customer_type",
    "adr", "required_car_parking_spaces", "total_of_special_requests",
    "reservation_status", "reservation_status_date",
]


def load_raw() -> pd.DataFrame:
    log.info("Loading raw CSV from %s", RAW_S3)
    df = pd.read_csv(RAW_S3)
    log.info("Loaded shape=%s", df.shape)
    return df


def validate(df: pd.DataFrame) -> bool:
    """Run Great Expectations suite. Returns True if all passed."""
    log.info("Setting up ephemeral Great Expectations context")
    context = gx.get_context(mode="ephemeral")

    ds = context.data_sources.add_pandas(name="pandas")
    asset = ds.add_dataframe_asset(name="hotels")
    batch_def = asset.add_batch_definition_whole_dataframe("whole")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})

    checks = [
        gxe.ExpectTableRowCountToBeBetween(min_value=80_000, max_value=200_000),
        gxe.ExpectTableColumnsToMatchOrderedList(column_list=EXPECTED_COLUMNS),
        gxe.ExpectColumnValuesToNotBeNull(column="is_canceled"),
        gxe.ExpectColumnValuesToBeInSet(column="is_canceled", value_set=[0, 1]),
        gxe.ExpectColumnValuesToBeInSet(column="hotel", value_set=["City Hotel", "Resort Hotel"]),
        gxe.ExpectColumnValuesToBeBetween(column="lead_time", min_value=0, max_value=800),
        gxe.ExpectColumnValuesToBeBetween(column="adr", min_value=-1000, max_value=10000),
        gxe.ExpectColumnValuesToNotBeNull(column="hotel"),
        gxe.ExpectColumnValuesToNotBeNull(column="lead_time"),
    ]

    all_ok = True
    for check in checks:
        res = batch.validate(check)
        name = type(check).__name__
        if res.success:
            log.info("  PASS  %s", name)
        else:
            log.error("  FAIL  %s  result=%s", name, res.result)
            all_ok = False
    return all_ok


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Light feature engineering. Drop target-leaking columns."""
    log.info("Transforming")
    out = df.copy()

    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
        "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
    }
    out["arrival_month_num"] = out["arrival_date_month"].map(month_map)
    out["arrival_date"] = pd.to_datetime({
        "year": out["arrival_date_year"],
        "month": out["arrival_month_num"],
        "day": out["arrival_date_day_of_month"],
    })

    # Derived features
    out["total_guests"] = out["adults"].fillna(0) + out["children"].fillna(0) + out["babies"].fillna(0)
    out["total_nights"] = out["stays_in_weekend_nights"] + out["stays_in_week_nights"]

    # Critical: drop columns that leak the target
    #   reservation_status: "Canceled" / "Check-Out" / "No-Show" — direct leak
    #   reservation_status_date: also post-hoc
    out = out.drop(columns=["reservation_status", "reservation_status_date"])

    log.info("Transformed shape=%s", out.shape)
    return out


def write(df: pd.DataFrame) -> None:
    log.info("Writing processed Parquet to %s", PROCESSED_S3)
    df.to_parquet(PROCESSED_S3, index=False, compression="snappy")
    log.info("Wrote %d rows", len(df))


def main() -> int:
    df = load_raw()

    if not validate(df):
        log.error("Validation failed — refusing to write processed data")
        return 1

    out = transform(df)
    write(out)
    log.info("ETL complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
