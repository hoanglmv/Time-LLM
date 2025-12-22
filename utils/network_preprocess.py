import pandas as pd
import os

# =================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# =================================================================
input_file = 'dataset/network/kpi_15_mins_3_months.csv'
output_dir = 'dataset/network/'

# Tạo thư mục đầu ra nếu chưa tồn tại
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"✅ Đã tạo thư mục: {output_dir}")

def preprocess_network_data():
    # 2. ĐỌC DỮ LIỆU
    print(f"📖 Đang đọc file: {input_file} ...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file {input_file}")
        return

    # Đổi tên timestamp thành date để khớp với yêu cầu của Model
    if 'timestamp' in df.columns:
        df.rename(columns={'timestamp': 'date'}, inplace=True)
    
    # Chuyển cột date sang định dạng datetime (bắt buộc để xử lý chuỗi thời gian)
    df['date'] = pd.to_datetime(df['date'])

    # Xác định tên cột enodeB (đề phòng viết hoa/thường)
    enodeb_col_name = 'enodeb' if 'enodeb' in df.columns else 'enodeB'
    
    # Lấy danh sách các cell duy nhất
    cell_names = df['cell_name'].unique()
    print(f"🔍 Tìm thấy {len(cell_names)} cells duy nhất.")

    # 3. XỬ LÝ TỪNG CELL
    for cell_name in cell_names:
        # Lọc dữ liệu cho cell hiện tại
        cell_df = df[df['cell_name'] == cell_name].copy()

        # Lấy tên enodeB trước khi xóa cột
        current_enodeb = "Unknown"
        if enodeb_col_name in cell_df.columns:
            current_enodeb = str(cell_df[enodeb_col_name].iloc[0])

        # Loại bỏ các cột không cần thiết cho huấn luyện
        cols_to_drop = ['cell_name', 'enodeb', 'enodeB']
        cell_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        # --- BƯỚC QUAN TRỌNG: XỬ LÝ TRÙNG LẶP (FIX LỖI VALUERROR) ---
        # Sắp xếp theo thời gian
        cell_df.sort_values(by='date', inplace=True)
        
        # Nếu có nhiều dòng trùng mốc thời gian, lấy giá trị trung bình (mean)
        cell_df = cell_df.groupby('date').mean().reset_index()

        # --- BƯỚC QUAN TRỌNG: ĐIỀN KHUYẾT CHUỖI THỜI GIAN ---
        # Đặt date làm index để resample
        cell_df.set_index('date', inplace=True)

        # Tạo khung thời gian 15 phút liên tục (không còn bị nhảy cóc)
        cell_df = cell_df.resample('15min').asfreq()

        # Nội suy tuyến tính để điền giá trị vào các mốc thời gian bị thiếu (NaN)
        cell_df = cell_df.interpolate(method='linear', limit_direction='both')
        
        # Điền nốt các giá trị ở cực đầu/cuối nếu vẫn còn trống
        cell_df = cell_df.ffill().bfill()

        # Đưa cột date quay trở lại
        cell_df.reset_index(inplace=True)

        # 4. LƯU FILE
        file_name = f"{current_enodeb}_{cell_name}.csv"
        output_path = os.path.join(output_dir, file_name)
        cell_df.to_csv(output_path, index=False)

        print(f"✨ Đã xử lý xong: {file_name} (Dòng: {len(cell_df)})")

    print("\n🚀 TẤT CẢ DỮ LIỆU ĐÃ ĐƯỢC CHUẨN HÓA VÀ SẴN SÀNG ĐỂ TRAIN!")

if __name__ == "__main__":
    preprocess_network_data()