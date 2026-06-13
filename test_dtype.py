import pandas as pd
import numpy as np
from missing_value_filler import MissingValueFiller, fill_missing_values


def test_dtype_preservation():
    """测试数据类型保持功能。"""

    print("=" * 60)
    print("测试场景 1：fit 用无缺失值的整数列，transform 用有缺失值的数据")
    print("=" * 60)

    train_data = pd.DataFrame({
        "value": [1, 2, 3, 4, 5],
    })
    print(f"fit 数据类型: {train_data['value'].dtype}")

    filler = MissingValueFiller(strategy="mean")
    filler.fit(train_data)
    print(f"填充值: {filler.fill_values}")

    test_data = pd.DataFrame({
        "value": [1, np.nan, 3, np.nan, 5],
    })
    print(f"transform 前数据类型: {test_data['value'].dtype}")

    result = filler.transform(test_data)
    print(f"transform 后数据类型: {result['value'].dtype}")
    print(f"结果:\n{result}")

    assert result["value"].dtype == train_data["value"].dtype, (
        f"数据类型不匹配: 期望 {train_data['value'].dtype}, 实际 {result['value'].dtype}"
    )
    print("✓ 测试通过：数据类型保持一致")

    print("\n" + "=" * 60)
    print("测试场景 2：pandas 可空整数类型 Int64")
    print("=" * 60)

    data_int64 = pd.DataFrame({
        "count": pd.array([3, 5, pd.NA, 7, 2, pd.NA, 9, 4], dtype=pd.Int64Dtype()),
    })
    print(f"原始数据类型: {data_int64['count'].dtype}")

    result_int64 = MissingValueFiller(strategy="median").fit_transform(data_int64)
    print(f"填充后数据类型: {result_int64['count'].dtype}")
    print(f"结果:\n{result_int64}")

    assert result_int64["count"].dtype == pd.Int64Dtype(), (
        f"数据类型不匹配: 期望 Int64, 实际 {result_int64['count'].dtype}"
    )
    print("✓ 测试通过：Int64 类型保持一致")

    print("\n" + "=" * 60)
    print("测试场景 3：浮点数类型保持不变")
    print("=" * 60)

    data_float = pd.DataFrame({
        "price": [1.5, 2.3, np.nan, 4.7, np.nan],
    })
    print(f"原始数据类型: {data_float['price'].dtype}")

    result_float = MissingValueFiller(strategy="mean").fit_transform(data_float)
    print(f"填充后数据类型: {result_float['price'].dtype}")
    print(f"结果:\n{result_float}")

    assert result_float["price"].dtype == np.float64, (
        f"数据类型不匹配: 期望 float64, 实际 {result_float['price'].dtype}"
    )
    print("✓ 测试通过：float64 类型保持一致")

    print("\n" + "=" * 60)
    print("测试场景 4：众数填充时整数类型保持")
    print("=" * 60)

    data_mode = pd.DataFrame({
        "category": pd.array([1, 2, 2, pd.NA, 1, pd.NA, 2, 3], dtype=pd.Int32Dtype()),
    })
    print(f"原始数据类型: {data_mode['category'].dtype}")

    result_mode = MissingValueFiller(strategy="mode").fit_transform(data_mode)
    print(f"填充后数据类型: {result_mode['category'].dtype}")
    print(f"结果:\n{result_mode}")

    assert result_mode["category"].dtype == pd.Int32Dtype(), (
        f"数据类型不匹配: 期望 Int32, 实际 {result_mode['category'].dtype}"
    )
    print("✓ 测试通过：众数填充时 Int32 类型保持一致")


def test_ffill_bfill():
    """测试前向填充和后向填充功能。"""

    print("\n" + "=" * 60)
    print("测试场景 5：前向填充 (ffill)")
    print("=" * 60)

    data = pd.DataFrame({
        "value": [1.0, np.nan, 3.0, np.nan, np.nan, 6.0],
        "count": pd.array([10, pd.NA, pd.NA, 40, pd.NA, 60], dtype=pd.Int64Dtype()),
        "label": ["A", pd.NA, "B", pd.NA, pd.NA, "C"],
    })
    print(f"原始数据:\n{data}")
    print(f"\n原始数据类型:\n{data.dtypes}")

    result_ffill = fill_missing_values(data, strategy="ffill")
    print(f"\n前向填充结果:\n{result_ffill}")
    print(f"\n填充后数据类型:\n{result_ffill.dtypes}")

    assert result_ffill.loc[1, "value"] == 1.0
    assert result_ffill.loc[3, "value"] == 3.0
    assert result_ffill.loc[4, "value"] == 3.0
    assert result_ffill.loc[1, "count"] == 10
    assert result_ffill.loc[2, "count"] == 10
    assert result_ffill.loc[4, "count"] == 40
    assert result_ffill.loc[1, "label"] == "A"
    assert result_ffill.loc[3, "label"] == "B"
    print("✓ 测试通过：前向填充值正确")

    assert result_ffill["value"].dtype == np.float64
    assert result_ffill["count"].dtype == pd.Int64Dtype()
    assert pd.api.types.is_string_dtype(result_ffill["label"])
    print("✓ 测试通过：前向填充后数据类型保持一致")

    print("\n" + "=" * 60)
    print("测试场景 6：后向填充 (bfill)")
    print("=" * 60)

    result_bfill = fill_missing_values(data, strategy="bfill")
    print(f"后向填充结果:\n{result_bfill}")
    print(f"\n填充后数据类型:\n{result_bfill.dtypes}")

    assert result_bfill.loc[1, "value"] == 3.0
    assert result_bfill.loc[3, "value"] == 6.0
    assert result_bfill.loc[4, "value"] == 6.0
    assert result_bfill.loc[1, "count"] == 40
    assert result_bfill.loc[2, "count"] == 40
    assert result_bfill.loc[4, "count"] == 60
    assert result_bfill.loc[1, "label"] == "B"
    assert result_bfill.loc[3, "label"] == "C"
    print("✓ 测试通过：后向填充值正确")

    assert result_bfill["value"].dtype == np.float64
    assert result_bfill["count"].dtype == pd.Int64Dtype()
    assert pd.api.types.is_string_dtype(result_bfill["label"])
    print("✓ 测试通过：后向填充后数据类型保持一致")

    print("\n" + "=" * 60)
    print("测试场景 7：ffill 对首行缺失值无法填充")
    print("=" * 60)

    data_first_nan = pd.DataFrame({
        "value": [np.nan, 2.0, 3.0],
    })
    result = fill_missing_values(data_first_nan, strategy="ffill")
    assert pd.isna(result.loc[0, "value"]), "前向填充首行为 NaN 时应保持不变"
    print("✓ 测试通过：首行缺失值 ffill 后仍为 NaN")

    print("\n" + "=" * 60)
    print("测试场景 8：bfill 对末行缺失值无法填充")
    print("=" * 60)

    data_last_nan = pd.DataFrame({
        "value": [1.0, 2.0, np.nan],
    })
    result = fill_missing_values(data_last_nan, strategy="bfill")
    assert pd.isna(result.loc[2, "value"]), "后向填充末行为 NaN 时应保持不变"
    print("✓ 测试通过：末行缺失值 bfill 后仍为 NaN")

    print("\n" + "=" * 60)
    print("测试场景 9：ffill/bfill 指定列填充")
    print("=" * 60)

    data_cols = pd.DataFrame({
        "a": [1.0, np.nan, 3.0],
        "b": [np.nan, 5.0, np.nan],
        "c": ["x", pd.NA, "z"],
    })
    result = fill_missing_values(data_cols, strategy="ffill", columns=["a", "c"])
    print(f"指定列 ffill 结果:\n{result}")

    assert result.loc[1, "a"] == 1.0
    assert pd.isna(result.loc[0, "b"]), "未指定的列 b 不应被填充"
    assert result.loc[1, "c"] == "x"
    print("✓ 测试通过：指定列前向填充正确")

    print("\n" + "=" * 60)
    print("测试场景 10：fit/transform 分离使用 ffill")
    print("=" * 60)

    train = pd.DataFrame({
        "value": [1.0, np.nan, 3.0],
    })
    test = pd.DataFrame({
        "value": [np.nan, 5.0, np.nan, 7.0],
    })

    filler = MissingValueFiller(strategy="ffill")
    filler.fit(train)
    result = filler.transform(test)
    print(f"fit 数据:\n{train}")
    print(f"transform 结果:\n{result}")

    assert result.loc[0, "value"] is pd.NA or pd.isna(result.loc[0, "value"])
    assert result.loc[1, "value"] == 5.0
    assert result.loc[2, "value"] == 5.0
    assert result.loc[3, "value"] == 7.0
    print("✓ 测试通过：fit/transform 分离时 ffill 正确")


if __name__ == "__main__":
    test_dtype_preservation()
    test_ffill_bfill()
    print("\n" + "=" * 60)
    print("所有测试通过！✓")
    print("=" * 60)
