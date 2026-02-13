#!/bin/bash

SERVICE_NAME="garbage_web.service"
HOTSPOT_SSID="Garbage_Hotspot"

echo "Stopping Web Control Service..."
systemctl stop $SERVICE_NAME
systemctl disable $SERVICE_NAME
echo "Service $SERVICE_NAME has been stopped and disabled from boot."

# Optional: Stop Hotspot
# echo "Stopping Hotspot..."
# nmcli connection down "$HOTSPOT_SSID"

echo "Done."
