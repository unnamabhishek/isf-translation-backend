#!/bin/bash
# Startup script for the translation agent
# This handles SSL certificate issues in corporate environments

echo "Starting Translation Agent with SSL bypass for corporate environments..."
echo ""

# Disable SSL verification for corporate proxy/firewall
export PYTHONHTTPSVERIFY=0
export CURL_CA_BUNDLE=""
export REQUESTS_CA_BUNDLE=""
export SSL_CERT_FILE=""
export NODE_TLS_REJECT_UNAUTHORIZED=0

# Show configuration
echo "Environment configured:"
echo "  - SSL verification disabled"
echo "  - Loading credentials from .env.local"
echo ""

# Check if .env.local exists
if [ ! -f .env.local ]; then
    echo "ERROR: .env.local not found!"
    echo "Please copy env-example.txt to .env.local and fill in your credentials"
    exit 1
fi

echo "Starting agent..."
echo "================================================================================"
python3 src/agent.py dev

