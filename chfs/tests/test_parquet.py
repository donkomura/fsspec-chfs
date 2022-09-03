import os
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def init():
    server = os.getenv("CHFS_SERVER")
    if not server:
        raise ValueError("CHFS server may not be up. Set CHFS_SERVER")
    yield

def test_parquet_write():
    df = pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})
    df.to_parquet('chfs://df1', engine='pyarrow')

def test_parquet_write_read():
    df = pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})
    df.to_parquet('chfs://df2', engine='pyarrow')
    df2 = pd.read_parquet('chfs://df2', engine='pyarrow')
    assert df.equals(df2)

def test_parquet_select():
    df = pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})
    df.to_parquet('chfs://df3', engine='pyarrow')
    df3 = pd.read_parquet('chfs://df3', engine='pyarrow')
    assert df[['col1']].equals(df3[['col1']])

