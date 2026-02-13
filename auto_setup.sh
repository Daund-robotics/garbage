#!/bin/bash

# Configuration
HOTSPOT_SSID="Garbage_Hotspot"
HOTSPOT_PASSWORD="password123"
APP_DIR="/home/rpi/garbage_4/garbage/web_control"
SERVICE_NAME="garbage_web.service"
USER_NAME=${SUDO_USER:-rpi}

echo "========================================="
echo "   Auto Setup: Hotspot & Web Control"
echo "========================================="

if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (sudo ./auto_setup.sh)"
  exit 1
fi

echo "[0/4] Cleaning up conflicts (dhcpcd/wpa_supplicant)..."
# Stop and disable dhcpcd if it exists, as it conflicts with NetworkManager hotspot
if systemctl list-unit-files | grep -q dhcpcd; then
    echo "Disabling dhcpcd to allow NetworkManager to manage Wi-Fi..."
    systemctl stop dhcpcd
    systemctl disable dhcpcd
fi

# Unblock Wi-Fi
if command -v rfkill &> /dev/null; then
    echo "Unblocking Wi-Fi..."
    rfkill unblock wifi
fi

echo "[1/4] Installing dependencies..."
apt-get update -y
apt-get install -y network-manager python3-flask python3-pip

# Ensure NetworkManager is running
systemctl enable NetworkManager
systemctl start NetworkManager

echo "[2/4] Configuring Wi-Fi Hotspot..."
# Remove existing connection if it exists to avoid duplicates
nmcli connection delete "$HOTSPOT_SSID" 2>/dev/null || true

# Create new hotspot connection
# 802-11-wireless.mode ap: Access Point mode
# ipv4.method shared: Share internet (or just act as a DHCP server)
echo "Creating Hotspot Connection..."
if nmcli con add type wifi ifname wlan0 con-name "$HOTSPOT_SSID" autoconnect yes ssid "$HOTSPOT_SSID"; then
    nmcli con modify "$HOTSPOT_SSID" 802-11-wireless.mode ap 802-11-wireless.band bg ipv4.method shared
    nmcli con modify "$HOTSPOT_SSID" wifi-sec.key-mgmt wpa-psk wifi-sec.psk "$HOTSPOT_PASSWORD"
    
    # Force bring up
    nmcli con up "$HOTSPOT_SSID"
    
    echo "Hotspot '$HOTSPOT_SSID' created with password '$HOTSPOT_PASSWORD'."
else
    echo "Error creating hotspot. ensure wlan0 is available."
    echo "Trying to list devices:"
    nmcli dev
fi

echo "[3/4] Setting up Web Control Service..."

# Create systemd service file
cat > /etc/systemd/system/$SERVICE_NAME <<EOF
[Unit]
Description=Garbage Robot Web Control Interface
After=network.target

[Service]
User=$USER_NAME
WorkingDirectory=$APP_DIR
ExecStart=/usr/bin/python3 $APP_DIR/app.py
Restart=always
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

# Reload daemon and enable service
systemctl daemon-reload
systemctl enable $SERVICE_NAME
systemctl restart $SERVICE_NAME

echo "Service '$SERVICE_NAME' enabled and started."

echo "[4/4] Finalizing..."
echo "---------------------------------------------------"
echo "Setup Complete!"
echo "1. Connect to Wi-Fi: $HOTSPOT_SSID"
echo "2. Password: $HOTSPOT_PASSWORD"
echo "3. Open Browser: http://10.42.0.1:5000"
echo "---------------------------------------------------"
echo "Note: Ensure your main script is located at:"
echo "/home/rpi/garbage_4/garbage/main_pi_ultra.py"
echo "---------------------------------------------------"
