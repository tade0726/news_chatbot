from abc import ABC, abstractmethod
from typing import override
import pandas as pd
from typing import Optional
import duckdb


class Dataset(ABC):
    @abstractmethod
    def read_data(self, data_path: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def write_data(self, data_path: str) -> None:
        pass


class DuckdbDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        df: Optional[pd.DataFrame] = None,
        table_name: str = "news",
        overwrite: bool = False,
    ) -> None:
        self.data_path = data_path
        self.df = df
        self.table_name = table_name
        self.overwrite = overwrite

    def write_data(self) -> None:

        if self.df is None:
            return

        df_to_write = self.df

        if self.overwrite:
            with duckdb.connect(self.data_path) as db:
                db.sql(f"DROP TABLE IF EXISTS {self.table_name}")
                db.sql(
                    f"CREATE TABLE IF NOT EXISTS {self.table_name} AS SELECT * FROM df_to_write"
                )
        else:
            # create table if not exists or insert data if it does
            with duckdb.connect(self.data_path) as db:

                # Check if table exists
                table_exists = (
                    db.sql(
                        f"SELECT name FROM sqlite_master WHERE type='table' AND name='{self.table_name}'"
                    ).fetchone()
                    is not None
                )

                if not table_exists:
                    # Create table with data if it doesn't exist
                    db.sql(
                        f"CREATE TABLE {self.table_name} AS SELECT * FROM df_to_write"
                    )
                else:
                    # Only insert data if table already exists
                    db.sql(f"INSERT INTO {self.table_name} SELECT * FROM df_to_write")

    def read_data(self) -> pd.DataFrame:
        with duckdb.connect(self.data_path) as db:
            # find the last batch query record
            latest_batch = db.sql(
                f"SELECT DISTINCT batch_query_time FROM {self.table_name} ORDER BY batch_query_time DESC LIMIT 1"
            ).fetchdf()

            if not latest_batch.empty:
                latest_batch_time = latest_batch.iloc[0]["batch_query_time"]
                # get all records from the latest batch
                return db.sql(
                    f"SELECT * FROM {self.table_name} WHERE batch_query_time = '{latest_batch_time}'"
                ).fetchdf()
            else:
                return pd.DataFrame()  # Return empty DataFrame if no records exist
