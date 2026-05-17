import pandas as pd

def aggregate_excel_data(file_path):
    df = pd.read_excel(file_path)
    
    # 特定の列の合計値を取得
    total_sales = df['売上'].sum()
    print(f"総売上: {total_sales}円")

# 使用例
aggregate_excel_data('/path/to/your/sales_data.xlsx')
