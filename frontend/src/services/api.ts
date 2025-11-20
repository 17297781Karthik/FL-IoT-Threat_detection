// API Service for FL-IoT-Threat Detection Dashboard

import axios from 'axios';
import type { AxiosInstance } from 'axios';
import type {
  Model,
  ModelInfo,
  LogEntry,
  TrainingMetrics,
  DetectionResult,
  Event,
  Config,
  Stats,
  LogLevel,
  LogType,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api';

class ApiService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  // Health Check
  async healthCheck(): Promise<{ status: string; timestamp: string; version: string }> {
    const response = await this.client.get('/health');
    return response.data;
  }

  // Models
  async getModels(): Promise<{ models: Model[]; count: number }> {
    const response = await this.client.get('/models');
    return response.data;
  }

  async getModelInfo(modelName: string): Promise<ModelInfo> {
    const response = await this.client.get(`/models/${modelName}/info`);
    return response.data;
  }

  // Logs
  async getLogs(
    type: LogType = 'server',
    level: LogLevel = 'all',
    limit: number = 100
  ): Promise<{ logs: LogEntry[]; count: number; type: string; level: string }> {
    const response = await this.client.get('/logs', {
      params: { type, level, limit },
    });
    return response.data;
  }

  // Training Metrics
  async getTrainingMetrics(): Promise<TrainingMetrics> {
    const response = await this.client.get('/metrics/training');
    return response.data;
  }

  // Detection Metrics
  async getDetectionMetrics(): Promise<{
    total_files: number;
    threats_detected: number;
    benign_files: number;
    results: DetectionResult[];
  }> {
    const response = await this.client.get('/metrics/detection');
    return response.data;
  }

  async getDetectionResults(): Promise<{ results: DetectionResult[]; count: number }> {
    const response = await this.client.get('/detection/results');
    return response.data;
  }

  // Events
  async getEvents(): Promise<{ events: Event[]; count: number }> {
    const response = await this.client.get('/events');
    return response.data;
  }

  // Configuration
  async getConfig(): Promise<Config> {
    const response = await this.client.get('/config');
    return response.data;
  }

  async updateConfig(config: Config): Promise<{ status: string; message: string }> {
    const response = await this.client.post('/config', config);
    return response.data;
  }

  // Statistics
  async getStats(): Promise<Stats> {
    const response = await this.client.get('/stats');
    return response.data;
  }
}

export const apiService = new ApiService();
export default apiService;
