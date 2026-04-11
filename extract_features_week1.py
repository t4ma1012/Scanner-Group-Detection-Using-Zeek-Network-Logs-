import pandas as pd
import numpy as np
import sys

def parse_conn_log(logfile):
    rows = []
    col_names = None
    with open(logfile, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#fields'):
                col_names = line.split('\t')[1:]
                continue
            if line.startswith('#'):
                continue
            parts = line.split('\t')
            if col_names and len(parts) == len(col_names):
                rows.append(parts)
    if col_names is None:
        print("❌ Error: No '#fields' header found in conn.log")
        return None
    if len(rows) == 0:
        print("❌ Error: No data rows found in conn.log")
        return None
    df = pd.DataFrame(rows, columns=col_names)
    df.rename(columns={'id.orig_h':'orig_h','id.orig_p':'orig_p',
                       'id.resp_h':'resp_h','id.resp_p':'resp_p'}, inplace=True)
    df['ts']         = pd.to_numeric(df['ts'],         errors='coerce')
    df['orig_p']     = pd.to_numeric(df['orig_p'],     errors='coerce')
    df['resp_p']     = pd.to_numeric(df['resp_p'],     errors='coerce')
    df['duration']   = pd.to_numeric(df['duration'],   errors='coerce').fillna(0)
    df['orig_bytes'] = pd.to_numeric(df['orig_bytes'], errors='coerce').fillna(0)
    df['resp_bytes'] = pd.to_numeric(df['resp_bytes'], errors='coerce').fillna(0)
    df.dropna(subset=['ts'], inplace=True)
    print(f"[✓] Parsed: {len(df)} rows")
    return df

def extract_features(df, window_size=30):
    min_ts = df['ts'].min()
    max_ts = df['ts'].max()
    print(f"[*] Time range: {(max_ts-min_ts)/60:.1f} minutes")
    features_list = []
    window_start = int(min_ts)
    window_count = 0
    while window_start < max_ts:
        window_end = window_start + window_size
        w = df[(df['ts'] >= window_start) & (df['ts'] < window_end)]
        if len(w) == 0:
            window_start += window_size
            continue
        window_count += 1
        features_list.append({
            'window_id'       : window_count,
            'window_start'    : window_start,
            'packet_count'    : len(w),
            'unique_src_ips'  : w['orig_h'].nunique(),
            'unique_dst_ips'  : w['resp_h'].nunique(),
            'unique_dst_ports': w['resp_p'].nunique(),
            'avg_duration'    : round(w['duration'].mean(), 4),
            'total_orig_bytes': int(w['orig_bytes'].sum()),
            'total_resp_bytes': int(w['resp_bytes'].sum()),
            'proto_tcp'       : (w['proto'] == 'tcp').sum(),
            'proto_udp'       : (w['proto'] == 'udp').sum(),
            'proto_icmp'      : (w['proto'] == 'icmp').sum(),
            'low_port_ratio'  : round((w['resp_p'] < 1024).sum() / len(w), 4),
        })
        window_start += window_size
    result = pd.DataFrame(features_list)
    print(f"[✓] Extracted: {len(result)} windows")
    return result

if __name__ == '__main__':
    try:
        print("[*] ===== ZEEK FEATURE EXTRACTION =====")
        df = parse_conn_log('conn.log')
        if df is None or df.empty:
            print("❌ Failed to parse conn.log")
            sys.exit(1)
        features = extract_features(df, window_size=30)
        features.to_csv('features.csv', index=False)
        print(f"\n[✅] Saved: features.csv")
        print(f"[*] Shape: {features.shape}")
        print(f"\n[*] First 5 windows:\n{features.head(5).to_string()}\n")
        print(f"[*] Stats:\n{features[['packet_count','unique_dst_ports','proto_tcp']].describe()}")
    except FileNotFoundError:
        print("❌ conn.log not found!")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
