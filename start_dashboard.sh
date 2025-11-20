#!/bin/bash

# FL-IoT Threat Detection Dashboard Launch Script

echo "🚀 Starting FL-IoT Threat Detection Dashboard..."
echo ""

# Check if frontend is built
if [ ! -d "frontend/dist" ]; then
    echo "📦 Building frontend..."
    cd frontend
    npm run build
    cd ..
    echo "✅ Frontend built successfully"
    echo ""
fi

# Start API server
echo "🔧 Starting API server on http://localhost:5000..."
python api_server.py

# Note: The API server will serve both the API endpoints and the frontend
# Access the dashboard at: http://localhost:5000
