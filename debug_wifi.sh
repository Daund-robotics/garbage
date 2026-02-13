#!/bin/bash

# Diagnostic script for Wi-Fi Hotspot Issues

echo "=== RPi Wi-Fi Debug Log ==="
date

echo -e "\n[1] Check NetworkManager Status"
systemctl status NetworkManager --no-pager

echo -e "\n[2] Check DHCPCD Status (Should be disabled)"
systemctl status dhcpcd --no-pager

echo -e "\n[3] List Network Devices"
nmcli dev

echo -e "\n[4] List Connections"
nmcli con show

echo -e "\n[5] Check IP Addresses"
ip addr

echo -e "\n[6] Check RFKill (Blocked?)"
rfkill list

echo -e "\n[7] Last 50 lines of System Log"
journalctl -xe | tail -n 50

echo "=== End Log ==="
