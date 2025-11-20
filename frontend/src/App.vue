<template>
  <div id="app">
    <nav class="navbar">
      <div class="nav-brand">
        <h1>FL-IoT Threat Detection</h1>
      </div>
      <div class="nav-links">
        <router-link to="/" class="nav-link">Dashboard</router-link>
        <router-link to="/models" class="nav-link">Models</router-link>
        <router-link to="/metrics" class="nav-link">Metrics</router-link>
        <router-link to="/events" class="nav-link">Events</router-link>
        <router-link to="/logs" class="nav-link">Logs</router-link>
        <router-link to="/config" class="nav-link">Configuration</router-link>
      </div>
      <div class="nav-status">
        <span :class="['status-indicator', healthStatus ? 'online' : 'offline']"></span>
        <span>{{ healthStatus ? 'Online' : 'Offline' }}</span>
      </div>
    </nav>
    
    <main class="main-content">
      <router-view />
    </main>
    
    <footer class="footer">
      <p>FL-IoT Threat Detection Dashboard v1.0.0 | Federated Learning for IoT Security</p>
    </footer>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { apiService } from './services/api';

const healthStatus = ref(false);

const checkHealth = async () => {
  try {
    await apiService.healthCheck();
    healthStatus.value = true;
  } catch (error) {
    healthStatus.value = false;
  }
};

onMounted(() => {
  checkHealth();
  // Check health every 30 seconds
  setInterval(checkHealth, 30000);
});
</script>

<style scoped>
#app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  background-color: #0f1419;
  color: #e4e6eb;
}

.navbar {
  background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
  padding: 1rem 2rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
}

.nav-brand h1 {
  margin: 0;
  font-size: 1.5rem;
  color: #ffffff;
  font-weight: 700;
}

.nav-links {
  display: flex;
  gap: 1.5rem;
}

.nav-link {
  color: #e0e7ff;
  text-decoration: none;
  padding: 0.5rem 1rem;
  border-radius: 0.375rem;
  transition: all 0.3s ease;
  font-weight: 500;
}

.nav-link:hover,
.nav-link.router-link-active {
  background-color: rgba(255, 255, 255, 0.1);
  color: #ffffff;
}

.nav-status {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: #e0e7ff;
  font-size: 0.875rem;
}

.status-indicator {
  width: 0.5rem;
  height: 0.5rem;
  border-radius: 50%;
  animation: pulse 2s infinite;
}

.status-indicator.online {
  background-color: #10b981;
  box-shadow: 0 0 8px #10b981;
}

.status-indicator.offline {
  background-color: #ef4444;
  box-shadow: 0 0 8px #ef4444;
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
}

.main-content {
  flex: 1;
  padding: 2rem;
  max-width: 1400px;
  margin: 0 auto;
  width: 100%;
}

.footer {
  background-color: #1a1f2e;
  padding: 1rem 2rem;
  text-align: center;
  color: #6b7280;
  font-size: 0.875rem;
  border-top: 1px solid #2d3748;
}

.footer p {
  margin: 0;
}

@media (max-width: 768px) {
  .navbar {
    flex-direction: column;
    gap: 1rem;
  }

  .nav-links {
    flex-wrap: wrap;
    justify-content: center;
  }

  .main-content {
    padding: 1rem;
  }
}
</style>
