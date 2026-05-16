import pandas as pd
import numpy as np
import os
import glob
from scipy.stats import entropy
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
DATA_DIR = "./raw" 
ZEEK_COLUMNS = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", 
    "proto", "service", "duration", "orig_bytes", "resp_bytes", 
    "conn_state", "local_orig", "local_resp", "missed_bytes", 
    "history", "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes", "tunnel_parents"
]

def calculate_entropy(series):
    value_counts = series.value_counts()
    return entropy(value_counts) if len(value_counts) > 0 else 0

def process_windows(df, label, tool_name):
    # Tiền xử lý: Ép kiểu thời gian và số học
    df['ts'] = pd.to_datetime(df['ts'], unit='s')
    df = df.replace('-', 0)
    
    for col in ['duration', 'orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Gom nhóm theo IP nguồn và Cửa sổ thời gian 60 giây
    grouped = df.groupby(['id.orig_h', pd.Grouper(key='ts', freq='60s')])
    features = []
    
    for (src_ip, time_window), group in grouped:
        if len(group) < 2: continue # Lọc bỏ nhiễu
        
        total_conn = len(group)
        
        # 1. NHÓM TỐC ĐỘ (Rate-based)
        conn_rate = total_conn / 60.0
        packets_per_conn = (group['orig_pkts'] + group['resp_pkts']).mean()
        bytes_per_conn = (group['orig_bytes'] + group['resp_bytes']).mean()
        
        # 2. NHÓM PHÂN TÁN (Entropy-based)
        port_entropy = calculate_entropy(group['id.resp_p'])
        ip_entropy = calculate_entropy(group['id.resp_h'])
        diff_dest_ports = group['id.resp_p'].nunique()
        
        # 3. NHÓM THỜI GIAN (Timing-based)
        iat = group['ts'].diff().dt.total_seconds().dropna()
        inter_arrival_time_std = iat.std() if len(iat) > 1 else 0
        duration_std = group['duration'].std()
        mean_duration = group['duration'].mean()
        
        # 4. NHÓM TRẠNG THÁI (State-based)
        syn_ratio = len(group[group['conn_state'] == 'S0']) / total_conn
        rst_ratio = len(group[group['conn_state'].str.contains('REJ|RST', na=False)]) / total_conn
        success_rate = len(group[group['conn_state'] == 'SF']) / total_conn
        
        # 5. NHÓM PAYLOAD / ỨNG DỤNG (Payload-based Alternative)
        # service != 0 (nghĩa là Zeek bắt được payload thực sự của Application layer)
        valid_payload_ratio = len(group[group['service'] != 0]) / total_conn
        # history == 'S' (Chỉ gửi SYN mà không làm gì khác)
        suspicious_history_ratio = len(group[group['history'] == 'S']) / total_conn

        # Đóng gói dữ liệu
        features.append({
            'conn_rate': conn_rate, 'packets_per_conn': packets_per_conn,
            'bytes_per_conn': bytes_per_conn, 'port_entropy': port_entropy,
            'ip_entropy': ip_entropy, 'diff_dest_ports': diff_dest_ports,
            'inter_arrival_time_std': inter_arrival_time_std, 'duration_std': duration_std,
            'mean_duration': mean_duration, 'syn_ratio': syn_ratio,
            'rst_ratio': rst_ratio, 'success_rate': success_rate,
            'valid_payload_ratio': valid_payload_ratio, 
            'suspicious_history_ratio': suspicious_history_ratio,
            'label': label,
            'tool': tool_name
        })
        
    return pd.DataFrame(features)

# ==========================================
# THỰC THI CHÍNH
# ==========================================
print("=== BẮT ĐẦU TRÍCH XUẤT ĐẶC TRƯNG TỪ ZEEK LOGS ===")
all_windows = []

# Duyệt từng thư mục tool
for tool_folder in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, tool_folder)
    if os.path.isdir(folder_path):
        # Đọc các file .gz trong thư mục
        for file in glob.glob(os.path.join(folder_path, "conn.log.gz")):
            try:
                # Đọc thẳng file nén gzip
                df = pd.read_csv(file, sep='\t', comment='#', header=None, names=ZEEK_COLUMNS, compression='gzip')
                if not df.empty:
                    # Gán nhãn: benign = 0 (bình thường), còn lại = 1 (tấn công)
                    label = 0 if tool_folder == 'benign' else 1
                    window_df = process_windows(df, label, tool_folder)
                    all_windows.append(window_df)
                    print(f" [+] Trích xuất {tool_folder}: tạo được {len(window_df)} dòng dữ liệu (cửa sổ 60s)")
            except Exception as e:
                print(f" [!] Bỏ qua file {file} do lỗi: {e}")

# Gom lại và xuất file
if all_windows:
    final_dataset = pd.concat(all_windows, ignore_index=True)
    final_dataset.fillna(0, inplace=True)
    
    # Lưu file
    output_name = "final_dataset_ML.csv"
    final_dataset.to_csv(output_name, index=False)
    
    print("\n=== HOÀN TẤT ===")
    print(f"Đã xuất file dữ liệu chuẩn: {output_name}")
    print(f"Tổng số mẫu dữ liệu mang đi huấn luyện: {len(final_dataset)} mẫu.")
else:
    print("\n[X] Lỗi: Không tìm thấy file log nào để trích xuất.")