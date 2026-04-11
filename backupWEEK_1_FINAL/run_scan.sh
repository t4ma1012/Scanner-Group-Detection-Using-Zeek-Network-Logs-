#!/bin/bash
echo "========== FULL MULTI-SCANNER SUITE (140+ min) ==========" > scan_timeline.txt
date >> scan_timeline.txt

# 8 rounds (tổng ~140 phút)
for i in $(seq 1 8); do
    echo "====== ROUND $i/8 ======" | tee -a scan_timeline.txt
    date >> scan_timeline.txt

    # Masscan: slower rates để chạy lâu hơn
    echo "[R$i] masscan rate=100..." | tee -a scan_timeline.txt
    sudo masscan 192.168.245.129 -p0-10000 --rate 100 > /dev/null 2>&1
    
    echo "[R$i] masscan rate=50..." | tee -a scan_timeline.txt
    sudo masscan 192.168.245.129 -p0-5000  --rate 50 > /dev/null 2>&1
    
    echo "[R$i] masscan rate=20..." | tee -a scan_timeline.txt
    sudo masscan 192.168.245.129 -p0-2000  --rate 20 > /dev/null 2>&1
    
    # nmap scans
    echo "[R$i] nmap -sS..." | tee -a scan_timeline.txt
    sudo nmap -sS -p 1-1000 192.168.245.129 > /dev/null 2>&1
    
    echo "[R$i] nmap -sV..." | tee -a scan_timeline.txt
    sudo nmap -sV -p 1-500  192.168.245.129 > /dev/null 2>&1
    
    echo "[R$i] nmap -O..." | tee -a scan_timeline.txt
    sudo nmap -O  -p 1-500  192.168.245.129 > /dev/null 2>&1
    
    # zmap
    echo "[R$i] zmap..." | tee -a scan_timeline.txt
    sudo zmap -p 22 192.168.245.129/32 > /dev/null 2>&1
    sudo zmap -p 80 192.168.245.129/32 > /dev/null 2>&1

    echo "✅ Round $i complete" >> scan_timeline.txt
    sleep 10  # Pause between rounds
done

echo "========== ALL ROUNDS COMPLETE ==========" >> scan_timeline.txt
date >> scan_timeline.txt
echo "Total duration should be ~140 minutes"
