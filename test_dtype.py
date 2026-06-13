import pandas as pd
import numpy as np
from missing_value_filler import MissingValueFiller


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

    print("\n" + "=" * 60)
    print("所有测试通过！✓")
    print("=" * 60)


if __name__ == "__main__":
    test_dtype_preservation()
