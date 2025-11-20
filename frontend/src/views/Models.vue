<template>
  <div class="models">
    <h1 class="page-title">Model Management</h1>
    
    <div v-if="loading" class="loading">
      <div class="spinner"></div>
    </div>
    
    <div v-else-if="error" class="error-message">
      {{ error }}
    </div>
    
    <div v-else>
      <div class="models-header">
        <p class="models-count">Total Models: <strong>{{ models.length }}</strong></p>
        <button @click="loadModels">Refresh</button>
      </div>
      
      <div v-if="models.length" class="models-grid">
        <div v-for="model in models" :key="model.name" class="model-card card">
          <div class="model-header">
            <h3 class="model-name">
              {{ model.name }}
              <span v-if="model.is_final" class="badge badge-success">Final</span>
            </h3>
          </div>
          
          <div class="model-details">
            <div class="detail-row">
              <span class="detail-label">Size:</span>
              <span class="detail-value">{{ formatSize(model.size) }}</span>
            </div>
            <div class="detail-row">
              <span class="detail-label">Created:</span>
              <span class="detail-value">{{ formatDate(model.created) }}</span>
            </div>
            <div class="detail-row">
              <span class="detail-label">Modified:</span>
              <span class="detail-value">{{ formatDate(model.modified) }}</span>
            </div>
          </div>
          
          <div class="model-actions">
            <button @click="viewModelInfo(model.name)" class="btn-info">View Details</button>
          </div>
        </div>
      </div>
      
      <div v-else class="no-data">
        No models found. Train your first model to see it here.
      </div>
      
      <!-- Model Info Modal -->
      <div v-if="selectedModelInfo" class="modal-overlay" @click="closeModal">
        <div class="modal" @click.stop>
          <div class="modal-header">
            <h2>{{ selectedModelInfo.name }}</h2>
            <button @click="closeModal" class="close-btn">×</button>
          </div>
          
          <div class="modal-body">
            <div class="info-section">
              <h3>Architecture</h3>
              <div class="arch-details">
                <div class="arch-row">
                  <span>Input Size:</span>
                  <span>{{ selectedModelInfo.architecture.input_size }}</span>
                </div>
                <div class="arch-row">
                  <span>Hidden Layers:</span>
                  <span>{{ selectedModelInfo.architecture.hidden_layers.join(' → ') }}</span>
                </div>
                <div class="arch-row">
                  <span>Output Size:</span>
                  <span>{{ selectedModelInfo.architecture.output_size }}</span>
                </div>
                <div class="arch-row">
                  <span>Total Parameters:</span>
                  <span>{{ selectedModelInfo.architecture.total_parameters.toLocaleString() }}</span>
                </div>
                <div class="arch-row">
                  <span>Trainable Parameters:</span>
                  <span>{{ selectedModelInfo.architecture.trainable_parameters.toLocaleString() }}</span>
                </div>
              </div>
            </div>
            
            <div class="info-section">
              <h3>File Information</h3>
              <div class="arch-details">
                <div class="arch-row">
                  <span>File Size:</span>
                  <span>{{ formatSize(selectedModelInfo.file_info.size) }}</span>
                </div>
                <div class="arch-row">
                  <span>Created:</span>
                  <span>{{ formatDate(selectedModelInfo.file_info.created) }}</span>
                </div>
                <div class="arch-row">
                  <span>Last Modified:</span>
                  <span>{{ formatDate(selectedModelInfo.file_info.modified) }}</span>
                </div>
              </div>
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
import type { Model, ModelInfo } from '../types';

const loading = ref(true);
const error = ref('');
const models = ref<Model[]>([]);
const selectedModelInfo = ref<ModelInfo | null>(null);

const loadModels = async () => {
  try {
    loading.value = true;
    error.value = '';
    const data = await apiService.getModels();
    models.value = data.models;
  } catch (err) {
    error.value = 'Failed to load models. Please ensure the API server is running.';
    console.error('Models error:', err);
  } finally {
    loading.value = false;
  }
};

const viewModelInfo = async (modelName: string) => {
  try {
    const info = await apiService.getModelInfo(modelName);
    selectedModelInfo.value = info;
  } catch (err) {
    console.error('Error loading model info:', err);
  }
};

const closeModal = () => {
  selectedModelInfo.value = null;
};

const formatSize = (bytes: number) => {
  const kb = bytes / 1024;
  const mb = kb / 1024;
  if (mb >= 1) {
    return `${mb.toFixed(2)} MB`;
  }
  return `${kb.toFixed(2)} KB`;
};

const formatDate = (dateString: string) => {
  try {
    return new Date(dateString).toLocaleString();
  } catch {
    return dateString;
  }
};

onMounted(() => {
  loadModels();
});
</script>

<style scoped>
.models {
  animation: fadeIn 0.5s ease-in;
}

.page-title {
  font-size: 2rem;
  font-weight: 700;
  margin-bottom: 2rem;
  color: #e4e6eb;
}

.models-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
}

.models-count {
  font-size: 1.125rem;
  color: #d1d5db;
}

.models-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
  gap: 1.5rem;
}

.model-card {
  transition: all 0.3s ease;
}

.model-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
}

.model-header {
  border-bottom: 1px solid #2d3748;
  padding-bottom: 1rem;
  margin-bottom: 1rem;
}

.model-name {
  font-size: 1.25rem;
  font-weight: 600;
  color: #3b82f6;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin: 0;
}

.model-details {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  margin-bottom: 1.5rem;
}

.detail-row {
  display: flex;
  justify-content: space-between;
  padding: 0.5rem 0;
}

.detail-label {
  color: #9ca3af;
  font-weight: 500;
}

.detail-value {
  color: #e4e6eb;
  font-weight: 600;
}

.model-actions {
  display: flex;
  gap: 0.5rem;
}

.btn-info {
  background-color: #3b82f6;
  flex: 1;
}

.btn-info:hover {
  background-color: #2563eb;
}

.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.75);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  animation: fadeIn 0.2s ease-in;
}

.modal {
  background-color: #1a1f2e;
  border-radius: 0.75rem;
  max-width: 600px;
  width: 90%;
  max-height: 80vh;
  overflow-y: auto;
  box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
  border: 1px solid #2d3748;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1.5rem;
  border-bottom: 1px solid #2d3748;
}

.modal-header h2 {
  margin: 0;
  color: #3b82f6;
  font-size: 1.5rem;
}

.close-btn {
  background: transparent;
  border: none;
  font-size: 2rem;
  color: #9ca3af;
  cursor: pointer;
  padding: 0;
  width: 2rem;
  height: 2rem;
  display: flex;
  align-items: center;
  justify-content: center;
}

.close-btn:hover {
  color: #e4e6eb;
  background-color: rgba(59, 130, 246, 0.1);
  border-radius: 0.375rem;
}

.modal-body {
  padding: 1.5rem;
}

.info-section {
  margin-bottom: 2rem;
}

.info-section:last-child {
  margin-bottom: 0;
}

.info-section h3 {
  font-size: 1.125rem;
  font-weight: 600;
  color: #e4e6eb;
  margin-bottom: 1rem;
}

.arch-details {
  background-color: rgba(59, 130, 246, 0.05);
  border-radius: 0.5rem;
  padding: 1rem;
}

.arch-row {
  display: flex;
  justify-content: space-between;
  padding: 0.5rem 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

.arch-row:last-child {
  border-bottom: none;
}

.arch-row span:first-child {
  color: #9ca3af;
  font-weight: 500;
}

.arch-row span:last-child {
  color: #e4e6eb;
  font-weight: 600;
}

.no-data {
  text-align: center;
  padding: 4rem 2rem;
  color: #6b7280;
}

@keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}
</style>
