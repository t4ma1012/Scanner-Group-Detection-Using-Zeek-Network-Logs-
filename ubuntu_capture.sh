#!/bin/bash
# ============================================================
# UBUNTU SIDE: Auto capture + save log per tool
# ============================================================

ZEEK_BIN="/opt/zeek/bin/zeek"
INTERFACE="ens33"
BASE_DIR="/home/t4ma/bigcat/scannergroup/data/raw"
SIGNAL_FILE="/tmp/scan_signal.txt"
LOG_FILE="/home/t4ma/bigcat/scannergroup/ubuntu_capture.log"
CHECKPOINT_FILE="/home/t4ma/bigcat/scannergroup/checkpoint.txt"

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_FILE"; }

mkdir -p "$BASE_DIR"/{nmap_sS,nmap_sV,nmap_sO,masscan_100,masscan_50,masscan_20,zmap_22,zmap_80,benign,scanners}

log "=== Ubuntu Capture Script Started ==="
log "Waiting for Kali to start..."

capture_tool() {
    local TOOL_NAME=$1
    local SAVE_DIR="$BASE_DIR/$TOOL_NAME"
    local ZEEK_PID=""

    log "--- START capture: $TOOL_NAME ---"

    TMPDIR=$(mktemp -d)
    cd "$TMPDIR" || exit

    sudo "$ZEEK_BIN" -C -i "$INTERFACE" local > /dev/null 2>&1 &
    ZEEK_PID=$!
    log "Zeek PID=$ZEEK_PID capturing for $TOOL_NAME"

    while true; do
        if [ -f "$SIGNAL_FILE" ]; then
            SIGNAL=$(sudo cat "$SIGNAL_FILE")
            if [ "$SIGNAL" = "${TOOL_NAME}_DONE" ]; then
                break
            fi
        fi
        sleep 1
    done

    sudo kill "$ZEEK_PID" 2>/dev/null
    sleep 2

    find "$TMPDIR" -type f ! -name "conn.log" -delete 
    TIMESTAMP=$(date +%s)
    cp "$TMPDIR"/conn.log "$SAVE_DIR/conn_${TIMESTAMP}.log" 2>/dev/null
    if [ -f "$SAVE_DIR/conn_${TIMESTAMP}.log" ]; then
        gzip -f "$SAVE_DIR/conn_${TIMESTAMP}.log"
        log "Saved: $SAVE_DIR/conn_${TIMESTAMP}.log.gz"
    fi

    cd /home/t4ma || exit
    rm -rf "$TMPDIR"
    
    sudo sh -c "echo '' > $SIGNAL_FILE"

    echo "$TOOL_NAME" >> "$CHECKPOINT_FILE"
    log "--- DONE capture: $TOOL_NAME [checkpoint saved] ---"
    sleep 3
}

sudo touch "$SIGNAL_FILE"
sudo chmod 666 "$SIGNAL_FILE"
sudo sh -c "echo '' > $SIGNAL_FILE"

# DANH SÁCH TOOLS CHỈ CÒN LẠI NMAP, ZMAP VÀ BENIGN
TOOLS=("nmap_sS" "nmap_sV" "nmap_sO" "zmap_22" "zmap_80" "benign")

log "=== Starting capture sequence (${#TOOLS[@]} tools) ==="

log "Waiting for Kali READY signal..."
while true; do
    if [ -f "$SIGNAL_FILE" ] && [ "$(sudo cat "$SIGNAL_FILE")" = "KALI_READY" ]; then
        break
    fi
    sleep 1
done
log "Kali is ready!"

sudo sh -c "echo 'UBUNTU_READY' > $SIGNAL_FILE"
log "Sent UBUNTU_READY signal."

for TOOL in "${TOOLS[@]}"; do
    if grep -qx "$TOOL" "$CHECKPOINT_FILE" 2>/dev/null; then
        log "--- SKIP (đã xong): $TOOL ---"
        continue
    fi
    capture_tool "$TOOL"
done

log "=== ALL CAPTURES COMPLETE ==="