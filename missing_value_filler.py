import pandas as pd
import numpy as np
from typing import Union, List, Optional


class MissingValueFiller:
    """缺失值填充服务类，支持均值、中位数、众数填充策略。"""

    def __init__(self, strategy: str = "mean"):
        """
        初始化缺失值填充服务。

        Args:
            strategy: 填充策略，可选值为 'mean'（均值）、'median'（中位数）、'mode'（众数）
        """
        valid_strategies = {"mean", "median", "mode"}
        if strategy not in valid_strategies:
            raise ValueError(
                f"无效的填充策略: {strategy}。可选策略: {', '.join(valid_strategies)}"
            )
        self.strategy = strategy
        self._fill_values = {}

    def fit(self, data: pd.DataFrame, columns: Optional[List[str]] = None) -> "MissingValueFiller":
        """
        根据数据计算填充值。

        Args:
            data: 输入数据 DataFrame
            columns: 需要填充的列名列表，若为 None 则处理所有数值型列

        Returns:
            self
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col not in data.columns:
                raise ValueError(f"列 {col} 不存在于数据中")

            if self.strategy == "mean":
                self._fill_values[col] = data[col].mean()
            elif self.strategy == "median":
                self._fill_values[col] = data[col].median()
            elif self.strategy == "mode":
                mode_result = data[col].mode()
                if len(mode_result) > 0:
                    self._fill_values[col] = mode_result.iloc[0]
                else:
                    self._fill_values[col] = np.nan

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        使用已计算的填充值填充数据中的缺失值。

        Args:
            data: 输入数据 DataFrame

        Returns:
            填充后的 DataFrame
        """
        if not self._fill_values:
            raise ValueError("尚未调用 fit 方法计算填充值")

        result = data.copy()
        for col, fill_value in self._fill_values.items():
            if col in result.columns:
                result[col] = result[col].fillna(fill_value)

        return result

    def fit_transform(self, data: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        先计算填充值，再填充缺失值。

        Args:
            data: 输入数据 DataFrame
            columns: 需要填充的列名列表，若为 None 则处理所有数值型列

        Returns:
            填充后的 DataFrame
        """
        return self.fit(data, columns).transform(data)

    @property
    def fill_values(self) -> dict:
        """获取各列的填充值。"""
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
        strategy: 填充策略，可选值为 'mean'、'median'、'mode'
        columns: 需要填充的列名列表，若为 None 则处理所有数值型列

    Returns:
        填充后的 DataFrame
    """
    filler = MissingValueFiller(strategy=strategy)
    return filler.fit_transform(data, columns)
