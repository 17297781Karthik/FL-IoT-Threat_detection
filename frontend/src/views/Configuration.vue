<template>
  <div class="configuration">
    <h1 class="page-title">Configuration</h1>
    
    <div v-if="loading" class="loading">
      <div class="spinner"></div>
    </div>
    
    <div v-else-if="error" class="error-message">
      {{ error }}
    </div>
    
    <div v-else>
      <div class="config-header">
        <p>Manage system configuration settings</p>
        <div class="button-group">
          <button @click="loadConfig" class="secondary">Refresh</button>
          <button @click="saveConfig" :disabled="!hasChanges">Save Changes</button>
        </div>
      </div>
      
      <div v-if="saveSuccess" class="success-message">
        Configuration saved successfully!
      </div>
      
      <!-- Paths Configuration -->
      <div class="card config-section">
        <div class="card-header">
          <h2 class="card-title">Paths</h2>
        </div>
        <div class="config-grid">
          <div class="config-field">
            <label>PCAP Directory</label>
            <input v-model="config.paths.pcap_directory" type="text" @input="markChanged" />
          </div>
          <div class="config-field">
            <label>Model Path</label>
            <input v-model="config.paths.model_path" type="text" @input="markChanged" />
          </div>
          <div class="config-field">
            <label>Results Directory</label>
            <input v-model="config.paths.results_directory" type="text" @input="markChanged" />
          </div>
          <div class="config-field">
            <label>Log File</label>
            <input v-model="config.paths.log_file" type="text" @input="markChanged" />
          </div>
        </div>
      </div>
      
      <!-- Model Configuration -->
      <div class="card config-section">
        <div class="card-header">
          <h2 class="card-title">Model Settings</h2>
        </div>
        <div class="config-grid">
          <div class="config-field">
            <label>Number of Features</label>
            <input v-model.number="config.model.num_features" type="number" @input="markChanged" />
          </div>
          <div class="config-field">
            <label>Number of Classes</label>
            <input v-model.number="config.model.num_classes" type="number" @input="markChanged" />
          </div>
        </div>
        
        <div class="config-field">
          <label>Attack Types</label>
          <div class="attack-types-list">
            <div v-for="(_name, id) in config.attack_types" :key="id" class="attack-type-item">
              <span class="attack-id">{{ id }}:</span>
              <input v-model="config.attack_types[id]" type="text" @input="markChanged" />
            </div>
          </div>
        </div>
      </div>
      
      <!-- Processing Configuration -->
      <div class="card config-section">
        <div class="card-header">
          <h2 class="card-title">Processing Settings</h2>
        </div>
        <div class="config-grid">
          <div class="config-field">
            <label>Batch Size</label>
            <input v-model.number="config.processing.batch_size" type="number" @input="markChanged" />
          </div>
          <div class="config-field">
            <label>Confidence Threshold</label>
            <input v-model.number="config.processing.confidence_threshold" type="number" step="0.1" min="0" max="1" @input="markChanged" />
          </div>
          <div class="config-field">
            <label>High Confidence Threshold</label>
            <input v-model.number="config.processing.high_confidence_threshold" type="number" step="0.1" min="0" max="1" @input="markChanged" />
          </div>
        </div>
      </div>
      
      <!-- Feature Extraction Configuration -->
      <div class="card config-section">
        <div class="card-header">
          <h2 class="card-title">Feature Extraction</h2>
        </div>
        <div class="config-grid">
          <div class="config-field">
            <label>Flow Timeout (seconds)</label>
            <input v-model.number="config.feature_extraction.flow_timeout" type="number" @input="markChanged" />
          </div>
          <div class="config-field">
            <label>Min Packets per Flow</label>
            <input v-model.number="config.feature_extraction.min_packets_per_flow" type="number" @input="markChanged" />
          </div>
          <div class="config-field">
            <label>Max Flows per File</label>
            <input v-model.number="config.feature_extraction.max_flows_per_file" type="number" @input="markChanged" />
          </div>
        </div>
      </div>
      
      <!-- Logging Configuration -->
      <div class="card config-section">
        <div class="card-header">
          <h2 class="card-title">Logging Settings</h2>
        </div>
        <div class="config-grid">
          <div class="config-field">
            <label>Log Level</label>
            <select v-model="config.logging.level" @change="markChanged">
              <option value="DEBUG">DEBUG</option>
              <option value="INFO">INFO</option>
              <option value="WARNING">WARNING</option>
              <option value="ERROR">ERROR</option>
            </select>
          </div>
          <div class="config-field">
            <label>Max Log Size</label>
            <input v-model="config.logging.max_log_size" type="text" @input="markChanged" />
          </div>
          <div class="config-field">
            <label>Backup Count</label>
            <input v-model.number="config.logging.backup_count" type="number" @input="markChanged" />
          </div>
        </div>
        
        <div class="config-grid">
          <div class="config-field checkbox-field">
            <label>
              <input v-model="config.logging.console_output" type="checkbox" @change="markChanged" />
              Console Output
            </label>
          </div>
          <div class="config-field checkbox-field">
            <label>
              <input v-model="config.logging.file_output" type="checkbox" @change="markChanged" />
              File Output
            </label>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { apiService } from '../services/api';
import type { Config } from '../types';

const loading = ref(true);
const error = ref('');
const saveSuccess = ref(false);
const hasChanges = ref(false);
const config = ref<Config>({
  paths: {
    pcap_directory: '',
    model_path: '',
    results_directory: '',
    log_file: '',
  },
  model: {
    num_features: 0,
    num_classes: 0,
    feature_names: [],
  },
  attack_types: {},
  processing: {
    lambda_values: [],
    batch_size: 0,
    confidence_threshold: 0,
    high_confidence_threshold: 0,
  },
  feature_extraction: {
    flow_timeout: 0,
    min_packets_per_flow: 0,
    max_flows_per_file: 0,
  },
  output: {
    save_features: false,
    save_predictions: false,
    generate_reports: false,
    output_format: '',
  },
  logging: {
    level: 'INFO',
    console_output: false,
    file_output: false,
    max_log_size: '',
    backup_count: 0,
  },
});

const loadConfig = async () => {
  try {
    loading.value = true;
    error.value = '';
    saveSuccess.value = false;
    const data = await apiService.getConfig();
    config.value = data;
    hasChanges.value = false;
  } catch (err) {
    error.value = 'Failed to load configuration. Please ensure the API server is running.';
    console.error('Config error:', err);
  } finally {
    loading.value = false;
  }
};

const saveConfig = async () => {
  try {
    await apiService.updateConfig(config.value);
    hasChanges.value = false;
    saveSuccess.value = true;
    setTimeout(() => {
      saveSuccess.value = false;
    }, 3000);
  } catch (err) {
    error.value = 'Failed to save configuration.';
    console.error('Save config error:', err);
  }
};

const markChanged = () => {
  hasChanges.value = true;
  saveSuccess.value = false;
};

onMounted(() => {
  loadConfig();
});
</script>

<style scoped>
.configuration {
  animation: fadeIn 0.5s ease-in;
}

.page-title {
  font-size: 2rem;
  font-weight: 700;
  margin-bottom: 2rem;
  color: #e4e6eb;
}

.config-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
}

.config-header p {
  color: #9ca3af;
}

.button-group {
  display: flex;
  gap: 0.75rem;
}

.success-message {
  background-color: rgba(16, 185, 129, 0.1);
  border: 1px solid #10b981;
  border-radius: 0.5rem;
  padding: 1rem;
  color: #10b981;
  margin-bottom: 1.5rem;
}

.config-section {
  margin-bottom: 2rem;
}

.config-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
  padding: 1rem 0;
}

.config-field {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.config-field label {
  color: #9ca3af;
  font-weight: 500;
  font-size: 0.875rem;
}

.config-field input[type="text"],
.config-field input[type="number"],
.config-field select {
  background-color: rgba(59, 130, 246, 0.05);
  color: #e4e6eb;
  border: 1px solid #2d3748;
  border-radius: 0.5rem;
  padding: 0.75rem;
  font-size: 1rem;
  transition: all 0.2s;
}

.config-field input[type="text"]:focus,
.config-field input[type="number"]:focus,
.config-field select:focus {
  outline: none;
  border-color: #3b82f6;
  background-color: rgba(59, 130, 246, 0.1);
}

.checkbox-field {
  flex-direction: row;
  align-items: center;
}

.checkbox-field label {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  cursor: pointer;
}

.checkbox-field input[type="checkbox"] {
  width: 1.25rem;
  height: 1.25rem;
  cursor: pointer;
}

.attack-types-list {
  display: grid;
  gap: 0.75rem;
  padding: 1rem;
  background-color: rgba(59, 130, 246, 0.05);
  border-radius: 0.5rem;
}

.attack-type-item {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.attack-id {
  color: #3b82f6;
  font-weight: 600;
  min-width: 2rem;
}

.attack-type-item input {
  flex: 1;
  background-color: #1a1f2e;
  color: #e4e6eb;
  border: 1px solid #2d3748;
  border-radius: 0.375rem;
  padding: 0.5rem;
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
  .config-header {
    flex-direction: column;
    gap: 1rem;
    align-items: flex-start;
  }
  
  .button-group {
    width: 100%;
  }
  
  .button-group button {
    flex: 1;
  }
}
</style>
