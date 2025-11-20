<template>
  <div class="metrics">
    <h1 class="page-title">Performance Metrics</h1>
    
    <div v-if="loading" class="loading">
      <div class="spinner"></div>
    </div>
    
    <div v-else-if="error" class="error-message">
      {{ error }}
    </div>
    
    <div v-else>
      <!-- Metrics Overview -->
      <div class="grid grid-cols-3" style="margin-bottom: 2rem;">
        <div class="metric-card card">
          <div class="metric-label">Total Training Rounds</div>
          <div class="metric-value">{{ trainingMetrics.rounds.length || 0 }}</div>
        </div>
        
        <div class="metric-card card">
          <div class="metric-label">Active Clients</div>
          <div class="metric-value">{{ trainingMetrics.clients.length || 0 }}</div>
        </div>
        
        <div class="metric-card card">
          <div class="metric-label">Detection Rate</div>
          <div class="metric-value">
            {{ detectionMetrics.total_files > 0 
              ? ((detectionMetrics.threats_detected / detectionMetrics.total_files) * 100).toFixed(1) 
              : 0 }}%
          </div>
        </div>
      </div>
      
      <!-- Detection Metrics -->
      <div class="card" style="margin-bottom: 2rem;">
        <div class="card-header">
          <h2 class="card-title">Detection Summary</h2>
        </div>
        <div class="detection-summary">
          <div class="summary-item">
            <div class="summary-label">Total Files Processed</div>
            <div class="summary-value">{{ detectionMetrics.total_files || 0 }}</div>
          </div>
          <div class="summary-item">
            <div class="summary-label">Threats Detected</div>
            <div class="summary-value threat">{{ detectionMetrics.threats_detected || 0 }}</div>
          </div>
          <div class="summary-item">
            <div class="summary-label">Benign Files</div>
            <div class="summary-value benign">{{ detectionMetrics.benign_files || 0 }}</div>
          </div>
        </div>
      </div>
      
      <!-- Recent Detection Results -->
      <div class="card">
        <div class="card-header">
          <h2 class="card-title">Recent Detection Results</h2>
        </div>
        <div v-if="detectionMetrics.results && detectionMetrics.results.length" class="results-table">
          <table>
            <thead>
              <tr>
                <th>PCAP File</th>
                <th>Prediction</th>
                <th>Confidence</th>
                <th>Timestamp</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(result, index) in detectionMetrics.results.slice(0, 15)" :key="index">
                <td>{{ getFileName(result.pcap_file) }}</td>
                <td>
                  <span :class="['badge', getThreatBadgeClass(result.prediction)]">
                    {{ result.prediction }}
                  </span>
                </td>
                <td>
                  <div class="confidence-bar">
                    <div class="confidence-fill" :style="{ width: (result.confidence * 100) + '%' }"></div>
                    <span class="confidence-text">{{ (result.confidence * 100).toFixed(1) }}%</span>
                  </div>
                </td>
                <td>{{ formatTimestamp(result.timestamp) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div v-else class="no-data">
          No detection results available
        </div>
      </div>
      
      <!-- Training Metrics Info -->
      <div class="card" style="margin-top: 2rem;">
        <div class="card-header">
          <h2 class="card-title">Training Information</h2>
        </div>
        <div class="training-info">
          <p v-if="trainingMetrics.rounds.length === 0" class="no-data">
            No training data available yet. Start a training session to see metrics.
          </p>
          <div v-else class="training-details">
            <div class="detail-item">
              <span class="detail-label">Completed Rounds:</span>
              <span class="detail-value">{{ trainingMetrics.rounds.join(', ') }}</span>
            </div>
            <div class="detail-item">
              <span class="detail-label">Participating Clients:</span>
              <span class="detail-value">{{ trainingMetrics.clients.length }} clients</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { apiService } from '../services/api';
import type { TrainingMetrics } from '../types';

const loading = ref(true);
const error = ref('');
const trainingMetrics = ref<TrainingMetrics>({
  rounds: [],
  accuracy: [],
  loss: [],
  clients: [],
});
const detectionMetrics = ref<any>({
  total_files: 0,
  threats_detected: 0,
  benign_files: 0,
  results: [],
});

const loadMetrics = async () => {
  try {
    loading.value = true;
    error.value = '';
    
    const [training, detection] = await Promise.all([
      apiService.getTrainingMetrics(),
      apiService.getDetectionMetrics(),
    ]);
    
    trainingMetrics.value = training;
    detectionMetrics.value = detection;
  } catch (err) {
    error.value = 'Failed to load metrics. Please ensure the API server is running.';
    console.error('Metrics error:', err);
  } finally {
    loading.value = false;
  }
};

const getThreatBadgeClass = (prediction: string) => {
  if (prediction === 'Benign') return 'badge-success';
  return 'badge-error';
};

const getFileName = (path: string) => {
  if (!path) return 'Unknown';
  return path.split('/').pop() || path;
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
  loadMetrics();
  // Refresh every 30 seconds
  setInterval(loadMetrics, 30000);
});
</script>

<style scoped>
.metrics {
  animation: fadeIn 0.5s ease-in;
}

.page-title {
  font-size: 2rem;
  font-weight: 700;
  margin-bottom: 2rem;
  color: #e4e6eb;
}

.metric-card {
  text-align: center;
  padding: 1.5rem;
}

.metric-label {
  color: #9ca3af;
  font-size: 0.875rem;
  margin-bottom: 0.5rem;
}

.metric-value {
  font-size: 2.5rem;
  font-weight: 700;
  color: #3b82f6;
}

.detection-summary {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 2rem;
  padding: 1rem;
}

.summary-item {
  text-align: center;
}

.summary-label {
  color: #9ca3af;
  font-size: 0.875rem;
  margin-bottom: 0.5rem;
}

.summary-value {
  font-size: 2rem;
  font-weight: 700;
  color: #3b82f6;
}

.summary-value.threat {
  color: #ef4444;
}

.summary-value.benign {
  color: #10b981;
}

.results-table {
  overflow-x: auto;
}

.confidence-bar {
  position: relative;
  width: 100%;
  height: 24px;
  background-color: rgba(59, 130, 246, 0.1);
  border-radius: 0.25rem;
  overflow: hidden;
}

.confidence-fill {
  position: absolute;
  top: 0;
  left: 0;
  height: 100%;
  background: linear-gradient(90deg, #3b82f6, #60a5fa);
  transition: width 0.3s ease;
}

.confidence-text {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  font-size: 0.75rem;
  font-weight: 600;
  color: #e4e6eb;
  z-index: 1;
}

.training-info {
  padding: 1rem;
}

.training-details {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.detail-item {
  display: flex;
  justify-content: space-between;
  padding: 0.75rem;
  background-color: rgba(59, 130, 246, 0.05);
  border-radius: 0.375rem;
}

.detail-label {
  color: #9ca3af;
  font-weight: 500;
}

.detail-value {
  color: #e4e6eb;
  font-weight: 600;
}

.no-data {
  text-align: center;
  padding: 2rem;
  color: #6b7280;
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

@media (max-width: 768px) {
  .detection-summary {
    grid-template-columns: 1fr;
  }
}
</style>
