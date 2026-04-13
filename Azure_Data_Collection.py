from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.monitor.query import LogsQueryClient
import time

start = time.time()

# Add workspace ID here
LOG_ANALYTICS_WORKSPACE_ID = "60c7f53e-249a-4077-b68e-55a4ae877d7c"

# Data fetch parameters.
DAYS_BACK = 90
CHUNK_DAYS = 0.125
PAUSE_SECONDS = 20  # <-- ADDED
OUTPUT_CSV = Path(f"DPE_concat_last_{DAYS_BACK}_days.csv")

# Use these columns to detect time column in each table, if needed.
TIME_COLUMNS_TO_TRY = ["Timestamp", "TimeGenerated"]

# Select tables or tables to export
FALLBACK_DEVICE_TABLES = [
    #"DeviceEvents",
    "DeviceProcessEvents",
    #"DeviceNetworkEvents",
    #"DeviceFileEvents",
    #"DeviceRegistryEvents",
    #"DeviceLogonEvents",
    #"DeviceInfo",
]


# Program starts here.
client = LogsQueryClient(credential=DefaultAzureCredential())


def _response_to_df(response) -> pd.DataFrame:
    if not response.tables:
        return pd.DataFrame()
    frames = []
    for t in response.tables:
        if t.rows:
            frames.append(pd.DataFrame(t.rows, columns=t.columns))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def discover_device_tables() -> List[str]:
    discovery_queries = [
        ".show tables | project TableName | where TableName startswith 'Device'",
        ".show tables | where TableName startswith 'Device' | project TableName",
    ]

    for q in discovery_queries:
        try:
            resp = client.query_workspace(
                workspace_id=LOG_ANALYTICS_WORKSPACE_ID,
                query=q,
                timespan=timedelta(hours=1),
            )
            df = _response_to_df(resp)
            if not df.empty:
                for col in ["TableName", "Name"]:
                    if col in df.columns:
                        tables = sorted(df[col].dropna().astype(str).unique().tolist())
                        if tables:
                            return tables
        except Exception:
            pass

    return FALLBACK_DEVICE_TABLES


def table_exists(table_name: str) -> bool:
    try:
        resp = client.query_workspace(
            workspace_id=LOG_ANALYTICS_WORKSPACE_ID,
            query=f"{table_name} | take 1",
            timespan=timedelta(days=1),
        )
        _ = _response_to_df(resp)
        return True
    except Exception:
        return False


def detect_time_column(table_name: str) -> Optional[str]:
    for time_col in TIME_COLUMNS_TO_TRY:
        try:
            resp = client.query_workspace(
                workspace_id=LOG_ANALYTICS_WORKSPACE_ID,
                query=f"{table_name} | take 1 | project {time_col}",
                timespan=timedelta(days=1),
            )
            _ = _response_to_df(resp)
            return time_col
        except Exception:
            continue
    return None


def export_all_device_tables_concat(
    days_back: int,
    chunk_days: float,
    output_csv: Path,
) -> pd.DataFrame:

    end_utc = datetime.now(timezone.utc)
    start_utc = end_utc - timedelta(days=days_back)

    candidate_tables = discover_device_tables()
    tables = [t for t in candidate_tables if table_exists(t)]

    all_frames: List[pd.DataFrame] = []

    for table in tables:
        time_col = detect_time_column(table)

        chunk_start = start_utc
        table_frames: List[pd.DataFrame] = []

        while chunk_start < end_utc:
            chunk_end = min(chunk_start + timedelta(days=chunk_days), end_utc)

            if time_col:
                kql = f"""
                {table}
                | where {time_col} between (datetime({chunk_start.isoformat()}) .. datetime({chunk_end.isoformat()}))
                """
                ts = (chunk_start, chunk_end)
            else:
                kql = f"{table}"
                ts = timedelta(days=1)

            try:
                resp = client.query_workspace(
                    workspace_id=LOG_ANALYTICS_WORKSPACE_ID,
                    query=kql,
                    timespan=ts,
                )
                df_chunk = _response_to_df(resp)

                if not df_chunk.empty:
                    df_chunk.insert(0, "SourceTable", table)
                    table_frames.append(df_chunk)

                print(f"{table}: {chunk_start.date()} → {chunk_end.date()} rows={len(df_chunk)}")

            except Exception as e:
                print(f"ERROR {table}: {chunk_start.date()} → {chunk_end.date()}: {e}")


            # 20 second pause to avoid throttling.
            print(f"Pausing {PAUSE_SECONDS} seconds to avoid throttling...")
            time.sleep(PAUSE_SECONDS)

            chunk_start = chunk_end

        if table_frames:
            df_table = pd.concat(table_frames, ignore_index=True, sort=True)
            df_table = df_table.drop_duplicates()
            all_frames.append(df_table)

    if not all_frames:
        empty = pd.DataFrame()
        empty.to_csv(output_csv, index=False, encoding="utf-8-sig")
        return empty

    df_all = pd.concat(all_frames, ignore_index=True, sort=True)
    df_all = df_all.drop_duplicates()

    df_all.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved CSV: {output_csv.resolve()}")
    print(f"Total rows exported: {len(df_all)}")

    return df_all

if __name__ == "__main__":
    export_all_device_tables_concat(
        days_back=DAYS_BACK,
        chunk_days=CHUNK_DAYS,
        output_csv=OUTPUT_CSV,
    )

    end = time.time()

    print(f"Execution time: {end - start:.2f} seconds")