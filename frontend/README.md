# FL-IoT Threat Detection Dashboard

A modern Vue.js + TypeScript dashboard for monitoring and managing the FL-IoT Threat Detection system.

## Features

- **Dashboard**: Real-time overview of system status, models, and detection metrics
- **Model Management**: Track and manage federated learning models with versioning
- **Metrics Analysis**: View training and detection metrics for both client and server
- **Events Timeline**: Track all system events with filtering capabilities
- **Logs Viewer**: Browse and filter system logs by type (server/client) and level (error/warn/info)
- **Configuration**: Manage system settings through an intuitive UI

## Tech Stack

- **Vue.js 3**: Progressive JavaScript framework
- **TypeScript**: Type-safe development
- **Vite**: Fast build tool and dev server
- **Axios**: HTTP client for API communication
- **Chart.js**: Data visualization
- **Vue Router**: Client-side routing

## Quick Start

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

```bash
# Install dependencies
npm install
```

### Development

```bash
# Start development server
npm run dev
```

The dashboard will be available at `http://localhost:5173`

### Production Build

```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

## API Configuration

The dashboard communicates with the backend API server. Configure the API URL in `.env`:

```env
VITE_API_URL=http://localhost:5000/api
```

## Project Structure

```
frontend/
├── src/
│   ├── components/     # Reusable Vue components
│   ├── views/          # Page components
│   │   ├── Dashboard.vue
│   │   ├── Models.vue
│   │   ├── Metrics.vue
│   │   ├── Events.vue
│   │   ├── Logs.vue
│   │   └── Configuration.vue
│   ├── services/       # API service layer
│   │   └── api.ts
│   ├── types/          # TypeScript type definitions
│   │   └── index.ts
│   ├── router/         # Vue Router configuration
│   │   └── index.ts
│   ├── App.vue         # Root component
│   ├── main.ts         # Application entry point
│   └── style.css       # Global styles
├── public/             # Static assets
├── .env                # Environment variables
├── index.html          # HTML entry point
├── vite.config.ts      # Vite configuration
└── package.json        # Project dependencies
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## License

Part of the FL-IoT-Threat_detection project.
