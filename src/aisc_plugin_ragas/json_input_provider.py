import io
import json
from typing import Any

import pandas as pd
from pandas import DataFrame

from aisc_plugin_interface.input_providers.base_input_provider import BaseInputProvider


class RawBytesProvider(BaseInputProvider):
    def _read_data(self, file_content: bytes) -> bytes:
        return file_content


class CsvToPandasProvider(BaseInputProvider):
    def _read_data(self, file_content: bytes) -> DataFrame:
        return pd.read_csv(io.BytesIO(file_content))


class JsonInputProvider(BaseInputProvider):
    def _read_data(self, file_content: bytes) -> dict[str, Any]:
        return json.loads(file_content.decode("utf-8"))
