import pandas as pd
import matplotlib.pyplot as plt

# 1. Đọc dữ liệu
df = pd.read_csv('./dataset/network/enodebA_enodebA1.csv')
target_col = 'ps_traffic_mb' # Cột bạn đang dự đoán

# 2. Chia index theo tỷ lệ mặc định của Time-LLM (thường là 70% Train, 10% Val, 20% Test)
n = len(df)
train_end = int(n * 0.7)
val_end = int(n * 0.8) # 70% + 10%

train_data = df[target_col].iloc[:train_end]
val_data = df[target_col].iloc[train_end:val_end]
test_data = df[target_col].iloc[val_end:]

# 3. Vẽ biểu đồ
plt.figure(figsize=(15, 6))
plt.plot(range(train_end), train_data, label='Train (Học)', color='blue', alpha=0.6)
plt.plot(range(train_end, val_end), val_data, label='Validation (Loss cao)', color='red', alpha=0.8)
plt.plot(range(val_end, n), test_data, label='Test (Loss thấp)', color='green', alpha=0.6)

plt.title(f'Phân bố dữ liệu: {target_col}')
plt.legend()
plt.grid(True)
plt.savefig('data_split_check.png')
print("Đã lưu biểu đồ kiểm tra tại data_split_check.png")