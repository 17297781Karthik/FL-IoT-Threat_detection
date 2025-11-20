// Type definitions for FL-IoT-Threat Detection Dashboard

export interface Model {
  name: string;
  path: string;
  size: number;
  created: string;
  modified: string;
  is_final: boolean;
}

export interface ModelInfo {
  name: string;
  architecture: {
    input_size: number;
    hidden_layers: number[];
    output_size: number;
    total_parameters: number;
    trainable_parameters: number;
  };
  file_info: {
    size: number;
    created: string;
    modified: string;
  };
}

export interface LogEntry {
  timestamp: string;
  logger: string;
  level: string;
  message: string;
}

export interface TrainingMetrics {
  rounds: number[];
  accuracy: number[];
  loss: number[];
  clients: string[];
}

export interface DetectionResult {
  filename: string;
  pcap_file: string;
  timestamp: string;
  prediction: string;
  confidence: number;
  features?: Record<string, number>;
}

export interface Event {
  timestamp: string;
  type: 'training' | 'client' | 'detection' | 'system';
  message: string;
  level: string;
}

export interface Config {
  paths: {
    pcap_directory: string;
    model_path: string;
    results_directory: string;
    log_file: string;
  };
  model: {
    num_features: number;
    num_classes: number;
    feature_names: string[];
  };
  attack_types: Record<string, string>;
  processing: {
    lambda_values: number[];
    batch_size: number;
    confidence_threshold: number;
    high_confidence_threshold: number;
  };
  feature_extraction: {
    flow_timeout: number;
    min_packets_per_flow: number;
    max_flows_per_file: number;
  };
  output: {
    save_features: boolean;
    save_predictions: boolean;
    generate_reports: boolean;
    output_format: string;
  };
  logging: {
    level: string;
    console_output: boolean;
    file_output: boolean;
    max_log_size: string;
    backup_count: number;
  };
}

export interface Stats {
  models: {
    total: number;
    latest: string | null;
  };
  training: {
    total_rounds: number;
    status: string;
  };
  detection: {
    total_processed: number;
    threats_detected: number;
  };
  logs: {
    server_size: number;
    client_size: number;
  };
}

export type LogLevel = 'all' | 'error' | 'warn' | 'info';
export type LogType = 'server' | 'client';
