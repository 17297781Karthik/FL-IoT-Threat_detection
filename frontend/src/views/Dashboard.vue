<template>
  <div class="dashboard">
    <h1 class="page-title">Dashboard</h1>
    
    <div v-if="loading" class="loading">
      <div class="spinner"></div>
    </div>
    
    <div v-else-if="error" class="error-message">
      {{ error }}
    </div>
    
    <div v-else>
      <!-- Stats Grid -->
      <div class="grid grid-cols-4" style="margin-bottom: 2rem;">
        <div class="stat-card card">
          <div class="stat-icon">📦</div>
          <div class="stat-value">{{ stats?.models.total || 0 }}</div>
          <div class="stat-label">Total Models</div>
        </div>
        
        <div class="stat-card card">
          <div class="stat-icon">🔄</div>
          <div class="stat-value">{{ stats?.training.total_rounds || 0 }}</div>
          <div class="stat-label">Training Rounds</div>
        </div>
        
        <div class="stat-card card">
          <div class="stat-icon">🔍</div>
          <div class="stat-value">{{ stats?.detection.total_processed || 0 }}</div>
          <div class="stat-label">Files Processed</div>
        </div>
        
        <div class="stat-card card">
          <div class="stat-icon">⚠️</div>
          <div class="stat-value">{{ stats?.detection.threats_detected || 0 }}</div>
          <div class="stat-label">Threats Detected</div>
        </div>
      </div>
      
      <!-- Charts and Info -->
      <div class="grid grid-cols-2">
        <!-- Latest Model Info -->
        <div class="card">
          <div class="card-header">
            <h2 class="card-title">Latest Model</h2>
          </div>
          <div v-if="stats?.models.latest" class="model-info">
            <div class="info-row">
              <span class="info-label">Model:</span>
              <span class="info-value">{{ stats.models.latest }}</span>
            </div>
            <div class="info-row">
              <span class="info-label">Status:</span>
              <span class="badge badge-success">Ready</span>
            </div>
            <div class="info-row">
              <span class="info-label">Training Status:</span>
              <span class="badge" :class="getTrainingStatusClass(stats.training.status)">
                {{ stats.training.status }}
              </span>
            </div>
          </div>
          <div v-else class="no-data">
            No models available
          </div>
        </div>
        
        <!-- Recent Events -->
        <div class="card">
          <div class="card-header">
            <h2 class="card-title">Recent Events</h2>
          </div>
          <div v-if="recentEvents.length" class="events-list">
            <div v-for="(event, index) in recentEvents.slice(0, 5)" :key="index" class="event-item">
              <span :class="['event-badge', `badge-${getLevelClass(event.level)}`]">
                {{ event.type }}
              </span>
              <span class="event-message">{{ truncate(event.message, 60) }}</span>
            </div>
          </div>
          <div v-else class="no-data">
            No recent events
          </div>
        </div>
      </div>
      
      <!-- Detection Results -->
      <div class="card" style="margin-top: 2rem;">
        <div class="card-header">
          <h2 class="card-title">Recent Detection Results</h2>
        </div>
        <div v-if="detectionResults.length" class="detection-table">
          <table>
            <thead>
              <tr>
                <th>File</th>
                <th>Prediction</th>
                <th>Confidence</th>
                <th>Timestamp</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(result, index) in detectionResults.slice(0, 10)" :key="index">
                <td>{{ result.filename }}</td>
                <td>
                  <span :class="['badge', result.prediction === 'Benign' ? 'badge-success' : 'badge-error']">
                    {{ result.prediction }}
                  </span>
                </td>
                <td>{{ (result.confidence * 100).toFixed(2) }}%</td>
                <td>{{ formatTimestamp(result.timestamp) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div v-else class="no-data">
          No detection results available
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { apiService } from '../services/api';
import type { Stats, Event, DetectionResult } from '../types';

const loading = ref(true);
const error = ref('');
const stats = ref<Stats | null>(null);
const recentEvents = ref<Event[]>([]);
const detectionResults = ref<DetectionResult[]>([]);

const loadDashboard = async () => {
  try {
    loading.value = true;
    error.value = '';
    
    const [statsData, eventsData, detectionData] = await Promise.all([
      apiService.getStats(),
      apiService.getEvents(),
      apiService.getDetectionResults(),
    ]);
    
    stats.value = statsData;
    recentEvents.value = eventsData.events;
    detectionResults.value = detectionData.results;
  } catch (err) {
    error.value = 'Failed to load dashboard data. Please ensure the API server is running.';
    console.error('Dashboard error:', err);
  } finally {
    loading.value = false;
  }
};

const getTrainingStatusClass = (status: string) => {
  const statusMap: Record<string, string> = {
    'running': 'badge-info',
    'idle': 'badge-success',
    'error': 'badge-error',
  };
  return statusMap[status.toLowerCase()] || 'badge-info';
};

const getLevelClass = (level: string) => {
  const levelMap: Record<string, string> = {
    'ERROR': 'error',
    'WARN': 'warning',
    'WARNING': 'warning',
    'INFO': 'info',
  };
  return levelMap[level.toUpperCase()] || 'info';
};

const truncate = (text: string, length: number) => {
  return text.length > length ? text.substring(0, length) + '...' : text;
};

const formatTimestamp = (timestamp: string) => {
  if (!timestamp) return 'N/A';
  try {
    return new Date(timestamp).toLocaleString();
  } catch {
    return timestamp;
  }
};

onMounted(() => {
  loadDashboard();
  // Refresh every 30 seconds
  setInterval(loadDashboard, 30000);
});
</script>

<style scoped>
.dashboard {
  animation: fadeIn 0.5s ease-in;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.page-title {
  font-size: 2rem;
  font-weight: 700;
  margin-bottom: 2rem;
  color: #e4e6eb;
}

.stat-card {
  text-align: center;
  transition: transform 0.2s ease;
}

.stat-card:hover {
  transform: translateY(-4px);
}

.stat-icon {
  font-size: 2.5rem;
  margin-bottom: 0.5rem;
}

.stat-value {
  font-size: 2rem;
  font-weight: 700;
  color: #3b82f6;
  margin-bottom: 0.25rem;
}

.stat-label {
  color: #9ca3af;
  font-size: 0.875rem;
}

.model-info,
.events-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.info-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem 0;
  border-bottom: 1px solid #2d3748;
}

.info-row:last-child {
  border-bottom: none;
}

.info-label {
  color: #9ca3af;
  font-weight: 500;
}

.info-value {
  color: #e4e6eb;
  font-weight: 600;
}

.event-item {
  display: flex;
  gap: 0.75rem;
  align-items: center;
  padding: 0.5rem;
  border-radius: 0.375rem;
  background-color: rgba(59, 130, 246, 0.05);
}

.event-badge {
  flex-shrink: 0;
  padding: 0.25rem 0.5rem;
  border-radius: 0.25rem;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
}

.event-message {
  color: #d1d5db;
  font-size: 0.875rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.no-data {
  text-align: center;
  padding: 2rem;
  color: #6b7280;
}

.detection-table {
  overflow-x: auto;
}
</style>
