<template>
  <div class="events">
    <h1 class="page-title">System Events</h1>
    
    <div v-if="loading" class="loading">
      <div class="spinner"></div>
    </div>
    
    <div v-else-if="error" class="error-message">
      {{ error }}
    </div>
    
    <div v-else>
      <div class="events-header">
        <p class="events-count">Total Events: <strong>{{ events.length }}</strong></p>
        <div class="filter-group">
          <select v-model="filterType" class="filter-select">
            <option value="all">All Types</option>
            <option value="training">Training</option>
            <option value="client">Client</option>
            <option value="detection">Detection</option>
            <option value="system">System</option>
          </select>
          <button @click="loadEvents">🔄 Refresh</button>
        </div>
      </div>
      
      <div class="card">
        <div v-if="filteredEvents.length" class="timeline">
          <div v-for="(event, index) in filteredEvents" :key="index" class="timeline-item">
            <div class="timeline-marker" :class="`marker-${getLevelClass(event.level)}`"></div>
            <div class="timeline-content">
              <div class="event-header">
                <span :class="['event-type-badge', `badge-${getTypeClass(event.type)}`]">
                  {{ event.type }}
                </span>
                <span class="event-time">{{ formatTimestamp(event.timestamp) }}</span>
              </div>
              <p class="event-message">{{ event.message }}</p>
            </div>
          </div>
        </div>
        <div v-else class="no-data">
          No events found
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';
import { apiService } from '../services/api';
import type { Event } from '../types';

const loading = ref(true);
const error = ref('');
const events = ref<Event[]>([]);
const filterType = ref<string>('all');

const filteredEvents = computed(() => {
  if (filterType.value === 'all') {
    return events.value;
  }
  return events.value.filter(event => event.type === filterType.value);
});

const loadEvents = async () => {
  try {
    loading.value = true;
    error.value = '';
    const data = await apiService.getEvents();
    events.value = data.events;
  } catch (err) {
    error.value = 'Failed to load events. Please ensure the API server is running.';
    console.error('Events error:', err);
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
  };
  return levelMap[level.toUpperCase()] || 'info';
};

const getTypeClass = (type: string) => {
  const typeMap: Record<string, string> = {
    'training': 'info',
    'client': 'success',
    'detection': 'warning',
    'system': 'error',
  };
  return typeMap[type] || 'info';
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
  loadEvents();
  // Refresh every 30 seconds
  setInterval(loadEvents, 30000);
});
</script>

<style scoped>
.events {
  animation: fadeIn 0.5s ease-in;
}

.page-title {
  font-size: 2rem;
  font-weight: 700;
  margin-bottom: 2rem;
  color: #e4e6eb;
}

.events-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
}

.events-count {
  font-size: 1.125rem;
  color: #d1d5db;
}

.filter-group {
  display: flex;
  gap: 1rem;
  align-items: center;
}

.filter-select {
  background-color: #1a1f2e;
  color: #e4e6eb;
  border: 1px solid #2d3748;
  border-radius: 0.5rem;
  padding: 0.5rem 1rem;
  font-size: 1rem;
  cursor: pointer;
}

.filter-select:focus {
  outline: none;
  border-color: #3b82f6;
}

.timeline {
  position: relative;
  padding: 1rem;
}

.timeline::before {
  content: '';
  position: absolute;
  left: 2rem;
  top: 0;
  bottom: 0;
  width: 2px;
  background: linear-gradient(to bottom, #3b82f6, #1e3a8a);
}

.timeline-item {
  position: relative;
  padding-left: 4rem;
  padding-bottom: 2rem;
}

.timeline-item:last-child {
  padding-bottom: 0;
}

.timeline-marker {
  position: absolute;
  left: 1.5rem;
  top: 0.5rem;
  width: 1rem;
  height: 1rem;
  border-radius: 50%;
  border: 3px solid #1a1f2e;
  z-index: 1;
}

.marker-info {
  background-color: #3b82f6;
  box-shadow: 0 0 8px #3b82f6;
}

.marker-success {
  background-color: #10b981;
  box-shadow: 0 0 8px #10b981;
}

.marker-warning {
  background-color: #fbbf24;
  box-shadow: 0 0 8px #fbbf24;
}

.marker-error {
  background-color: #ef4444;
  box-shadow: 0 0 8px #ef4444;
}

.timeline-content {
  background-color: rgba(59, 130, 246, 0.05);
  border-radius: 0.5rem;
  padding: 1rem;
  border-left: 3px solid #3b82f6;
}

.event-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.event-type-badge {
  padding: 0.25rem 0.75rem;
  border-radius: 0.25rem;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
}

.event-time {
  color: #9ca3af;
  font-size: 0.875rem;
}

.event-message {
  color: #d1d5db;
  margin: 0;
  line-height: 1.6;
}

.no-data {
  text-align: center;
  padding: 4rem 2rem;
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
  .events-header {
    flex-direction: column;
    gap: 1rem;
    align-items: flex-start;
  }
  
  .filter-group {
    width: 100%;
  }
  
  .filter-select {
    flex: 1;
  }
}
</style>
