<template>
  <div class="logs">
    <h1 class="page-title">System Logs</h1>
    
    <div v-if="loading" class="loading">
      <div class="spinner"></div>
    </div>
    
    <div v-else-if="error" class="error-message">
      {{ error }}
    </div>
    
    <div v-else>
      <div class="logs-controls">
        <div class="control-group">
          <label>Log Type:</label>
          <select v-model="logType" @change="loadLogs" class="control-select">
            <option value="server">Server</option>
            <option value="client">Client</option>
          </select>
        </div>
        
        <div class="control-group">
          <label>Level:</label>
          <select v-model="logLevel" @change="loadLogs" class="control-select">
            <option value="all">All</option>
            <option value="error">Error</option>
            <option value="warn">Warning</option>
            <option value="info">Info</option>
          </select>
        </div>
        
        <div class="control-group">
          <label>Limit:</label>
          <select v-model="logLimit" @change="loadLogs" class="control-select">
            <option :value="50">50</option>
            <option :value="100">100</option>
            <option :value="200">200</option>
            <option :value="500">500</option>
          </select>
        </div>
        
        <button @click="loadLogs" class="refresh-btn">Refresh</button>
      </div>
      
      <div class="card">
        <div class="card-header">
          <h2 class="card-title">
            {{ logType.charAt(0).toUpperCase() + logType.slice(1) }} Logs 
            ({{ logs.length }} entries)
          </h2>
        </div>
        
        <div v-if="logs.length" class="logs-container">
          <div v-for="(log, index) in logs" :key="index" class="log-entry" :class="`log-${getLevelClass(log.level)}`">
            <div class="log-header">
              <span :class="['log-level-badge', `badge-${getLevelClass(log.level)}`]">
                {{ log.level }}
              </span>
              <span class="log-timestamp">{{ log.timestamp }}</span>
            </div>
            <div class="log-body">
              <span class="log-logger">{{ log.logger }}:</span>
              <span class="log-message">{{ log.message }}</span>
            </div>
          </div>
        </div>
        
        <div v-else class="no-data">
          No logs available
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { apiService } from '../services/api';
import type { LogEntry, LogType, LogLevel } from '../types';

const loading = ref(true);
const error = ref('');
const logs = ref<LogEntry[]>([]);
const logType = ref<LogType>('server');
const logLevel = ref<LogLevel>('all');
const logLimit = ref(100);

const loadLogs = async () => {
  try {
    loading.value = true;
    error.value = '';
    const data = await apiService.getLogs(logType.value, logLevel.value, logLimit.value);
    logs.value = data.logs;
  } catch (err) {
    error.value = 'Failed to load logs. Please ensure the API server is running.';
    console.error('Logs error:', err);
  } finally {
    loading.value = false;
  }
};

const getLevelClass = (level: string) => {
  const levelMap: Record<string, string> = {
    'ERROR': 'error',
    'WARN': 'warning',
    'WARNING': 'warning',
    'INFO': 'info',
    'DEBUG': 'info',
  };
  return levelMap[level.toUpperCase()] || 'info';
};

onMounted(() => {
  loadLogs();
  // Auto-refresh every 10 seconds
  setInterval(loadLogs, 10000);
});
</script>

<style scoped>
.logs {
  animation: fadeIn 0.5s ease-in;
}

.page-title {
  font-size: 2rem;
  font-weight: 700;
  margin-bottom: 2rem;
  color: #e4e6eb;
}

.logs-controls {
  display: flex;
  gap: 1.5rem;
  align-items: flex-end;
  margin-bottom: 2rem;
  flex-wrap: wrap;
}

.control-group {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.control-group label {
  color: #9ca3af;
  font-size: 0.875rem;
  font-weight: 500;
}

.control-select {
  background-color: #1a1f2e;
  color: #e4e6eb;
  border: 1px solid #2d3748;
  border-radius: 0.5rem;
  padding: 0.5rem 1rem;
  font-size: 1rem;
  cursor: pointer;
}

.control-select:focus {
  outline: none;
  border-color: #3b82f6;
}

.refresh-btn {
  padding: 0.5rem 1rem;
}

.logs-container {
  max-height: 70vh;
  overflow-y: auto;
  padding: 1rem;
}

.log-entry {
  margin-bottom: 1rem;
  padding: 1rem;
  border-radius: 0.5rem;
  border-left: 4px solid;
  font-family: 'Monaco', 'Menlo', monospace;
  font-size: 0.875rem;
}

.log-info {
  background-color: rgba(59, 130, 246, 0.05);
  border-left-color: #3b82f6;
}

.log-warning {
  background-color: rgba(251, 191, 36, 0.05);
  border-left-color: #fbbf24;
}

.log-error {
  background-color: rgba(239, 68, 68, 0.05);
  border-left-color: #ef4444;
}

.log-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.log-level-badge {
  padding: 0.125rem 0.5rem;
  border-radius: 0.25rem;
  font-size: 0.75rem;
  font-weight: 700;
  text-transform: uppercase;
}

.log-timestamp {
  color: #6b7280;
  font-size: 0.75rem;
}

.log-body {
  color: #d1d5db;
  line-height: 1.6;
}

.log-logger {
  color: #9ca3af;
  font-weight: 600;
  margin-right: 0.5rem;
}

.log-message {
  color: #e4e6eb;
}

.no-data {
  text-align: center;
  padding: 4rem 2rem;
  color: #6b7280;
}

/* Custom scrollbar */
.logs-container::-webkit-scrollbar {
  width: 8px;
}

.logs-container::-webkit-scrollbar-track {
  background: #1a1f2e;
}

.logs-container::-webkit-scrollbar-thumb {
  background: #3b82f6;
  border-radius: 4px;
}

.logs-container::-webkit-scrollbar-thumb:hover {
  background: #2563eb;
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
  .logs-controls {
    flex-direction: column;
    align-items: stretch;
  }
  
  .control-group {
    width: 100%;
  }
}
</style>
