# FL-IoT Threat Detection Dashboard

Quick start guide for running the dashboard in development mode.

> **Note:** This frontend implementation is part of the fullstack branch.

## Development Mode

Run both the API server and frontend development server:

### Terminal 1 - API Server
```bash
python api_server.py
```

### Terminal 2 - Frontend Dev Server
```bash
cd frontend
npm run dev
```

Access the dashboard at: http://localhost:5173

## Production Mode

Build and serve the complete application:

```bash
# Build frontend
cd frontend
npm run build
cd ..

# Start API server (serves both API and frontend)
python api_server.py
```

Access the dashboard at: http://localhost:5000

## Quick Launch Script

Use the provided launch script:

```bash
chmod +x start_dashboard.sh
./start_dashboard.sh
```

## API Endpoints

The API server provides the following endpoints:

- `GET /api/health` - Health check
- `GET /api/models` - List all models
- `GET /api/models/{name}/info` - Get model details
- `GET /api/logs` - Get system logs (filterable)
- `GET /api/metrics/training` - Get training metrics
- `GET /api/metrics/detection` - Get detection metrics
- `GET /api/detection/results` - Get detection results
- `GET /api/events` - Get system events
- `GET /api/config` - Get configuration
- `POST /api/config` - Update configuration
- `GET /api/stats` - Get system statistics

## Features

### Dashboard
- Real-time system overview
- Model metrics and versioning
- Detection statistics
- Recent events timeline

### Models Page
- View all trained models
- Model details and architecture
- Version tracking

### Metrics Page
- Training performance metrics
- Detection success rates
- Detailed result tables

### Events Page
- System event timeline
- Filterable by type (training, client, detection, system)
- Real-time updates

### Logs Page
- Server and client logs
- Filterable by level (error, warn, info)
- Auto-refresh capability

### Configuration Page
- Manage paths and directories
- Model settings
- Processing parameters
- Feature extraction settings
- Logging configuration
