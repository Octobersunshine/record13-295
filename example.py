import pandas as pd
import numpy as np
from missing_value_filler import MissingValueFiller, fill_missing_values


def main():
    data = pd.DataFrame(
        {
            "age": [25, 30, np.nan, 35, 28, np.nan, 40, 32],
            "salary": [50000, 60000, 55000, np.nan, 65000, 70000, np.nan, 58000],
            "score": [85, 90, 78, 90, np.nan, 90, 88, np.nan],
            "count": pd.array([3, 5, np.nan, 7, 2, np.nan, 9, 4], dtype=pd.Int64Dtype()),
            "city": ["北京", "上海", "北京", np.nan, "深圳", "上海", "北京", "广州"],
        }
    )

    print("=" * 60)
    print("原始数据（含缺失值）:")
    print("=" * 60)
    print(data)
    print("\n各列数据类型:")
    print(data.dtypes)
    print("\n缺失值统计:")
    print(data.isnull().sum())

    print("\n" + "=" * 60)
    print("方式一：使用便捷函数 fill_missing_values")
    print("=" * 60)

    print("\n--- 均值填充 (mean) ---")
    result_mean = fill_missing_values(data, strategy="mean")
    print(result_mean)
    print("\n各列数据类型:")
    print(result_mean.dtypes)

    print("\n--- 中位数填充 (median) ---")
    result_median = fill_missing_values(data, strategy="median")
    print(result_median)
    print("\n各列数据类型:")
    print(result_median.dtypes)

    print("\n--- 众数填充 (mode) ---")
    result_mode = fill_missing_values(data, strategy="mode")
    print(result_mode)
    print("\n各列数据类型:")
    print(result_mode.dtypes)

    print("\n" + "=" * 60)
    print("方式二：使用 MissingValueFiller 类（支持 fit/transform 分离）")
    print("=" * 60)

    filler = MissingValueFiller(strategy="mean")
    filler.fit(data, columns=["age", "salary"])
    print(f"\n计算得到的填充值: {filler.fill_values}")

    result = filler.transform(data)
    print("\n填充后的数据:")
    print(result)
    print("\n各列数据类型:")
    print(result.dtypes)

    print("\n" + "=" * 60)
    print("仅填充指定列（age 和 score 使用中位数）")
    print("=" * 60)
    result_selected = fill_missing_values(data, strategy="median", columns=["age", "score"])
    print(result_selected)
    print("\n各列数据类型:")
    print(result_selected.dtypes)


if __name__ == "__main__":
    main()
