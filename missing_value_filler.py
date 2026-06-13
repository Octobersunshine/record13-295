import pandas as pd
import numpy as np
from typing import Union, List, Optional


class MissingValueFiller:
    """缺失值填充服务类，支持均值、中位数、众数、前向填充、后向填充策略。"""

    def __init__(self, strategy: str = "mean"):
        """
        初始化缺失值填充服务。

        Args:
            strategy: 填充策略，可选值为：
                'mean'（均值）、'median'（中位数）、'mode'（众数）、
                'ffill'（前向填充）、'bfill'（后向填充）
        """
        valid_strategies = {"mean", "median", "mode", "ffill", "bfill"}
        if strategy not in valid_strategies:
            raise ValueError(
                f"无效的填充策略: {strategy}。可选策略: {', '.join(valid_strategies)}"
            )
        self.strategy = strategy
        self._fill_values = {}
        self._dtypes = {}
        self._columns = []

    def fit(self, data: pd.DataFrame, columns: Optional[List[str]] = None) -> "MissingValueFiller":
        """
        根据数据计算填充值。

        对于 ffill/bfill 策略，仅记录列名和数据类型，不计算统一填充值。

        Args:
            data: 输入数据 DataFrame
            columns: 需要填充的列名列表，若为 None 则处理所有数值型列
                     （ffill/bfill 策略下默认处理所有列）

        Returns:
            self
        """
        if columns is None:
            if self.strategy in ("ffill", "bfill"):
                columns = data.columns.tolist()
            else:
                columns = data.select_dtypes(include=[np.number]).columns.tolist()

        self._columns = columns

        for col in columns:
            if col not in data.columns:
                raise ValueError(f"列 {col} 不存在于数据中")

            original_dtype = data[col].dtype
            self._dtypes[col] = original_dtype

            if self.strategy in ("ffill", "bfill"):
                continue

            if self.strategy == "mean":
                fill_value = data[col].mean()
            elif self.strategy == "median":
                fill_value = data[col].median()
            elif self.strategy == "mode":
                mode_result = data[col].mode()
                if len(mode_result) > 0:
                    fill_value = mode_result.iloc[0]
                else:
                    fill_value = np.nan

            if pd.api.types.is_integer_dtype(original_dtype):
                if pd.notna(fill_value):
                    fill_value = round(fill_value)

            self._fill_values[col] = fill_value

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        使用已计算的填充值填充数据中的缺失值。

        Args:
            data: 输入数据 DataFrame

        Returns:
            填充后的 DataFrame
        """
        if not self._columns:
            raise ValueError("尚未调用 fit 方法计算填充值")

        result = data.copy()

        for col in self._columns:
            if col not in result.columns:
                continue

            original_dtype = self._dtypes.get(col)

            if self.strategy == "ffill":
                result[col] = result[col].ffill()
            elif self.strategy == "bfill":
                result[col] = result[col].bfill()
            else:
                fill_value = self._fill_values.get(col)
                if fill_value is not None:
                    result[col] = result[col].fillna(fill_value)

            if original_dtype is not None and result[col].dtype != original_dtype:
                try:
                    result[col] = result[col].astype(original_dtype)
                except (ValueError, TypeError):
                    pass

        return result

    def fit_transform(self, data: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        先计算填充值，再填充缺失值。

        Args:
            data: 输入数据 DataFrame
            columns: 需要填充的列名列表，若为 None 则处理所有数值型列
                     （ffill/bfill 策略下默认处理所有列）

        Returns:
            填充后的 DataFrame
        """
        return self.fit(data, columns).transform(data)

    @property
    def fill_values(self) -> dict:
        """获取各列的填充值（ffill/bfill 策略下返回空字典）。"""
        return self._fill_values.copy()


def fill_missing_values(
    data: pd.DataFrame,
    strategy: str = "mean",
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    便捷函数：使用指定策略填充缺失值。

    Args:
        data: 输入数据 DataFrame
        strategy: 填充策略，可选值为 'mean'、'median'、'mode'、'ffill'、'bfill'
        columns: 需要填充的列名列表，若为 None 则处理所有数值型列
                 （ffill/bfill 策略下默认处理所有列）

    Returns:
        填充后的 DataFrame
    """
    filler = MissingValueFiller(strategy=strategy)
    return filler.fit_transform(data, columns)
