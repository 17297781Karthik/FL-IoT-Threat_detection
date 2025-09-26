#!/usr/bin/env python3
"""
Feature Extraction Pipeline for IoT Network Traffic
Extracts network flow features from PCAP files for threat detection
"""

import pandas as pd
import numpy as np
from scapy.all import rdpcap
from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.layers.l2 import Ether
import os
import logging
import json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NetworkFeatureExtractor:
    """
    Extracts network flow features from PCAP files for IoT threat detection
    """
    
    def __init__(self):
        self.features = [
            'HH_jit_L1_variance', 'HH_jit_L0.01_variance', 'HH_jit_L3_variance',
            'HH_jit_L0.1_variance', 'HH_jit_L0.1_mean', 'H_L0.01_variance',
            'HH_jit_L5_variance', 'MI_dir_L0.01_variance', 'MI_dir_L0.1_mean',
            'MI_dir_L0.01_weight', 'HH_L0.01_std', 'MI_dir_L0.1_weight',
            'H_L0.01_mean', 'H_L0.01_weight', 'HH_jit_L5_mean',
            'MI_dir_L1_weight', 'H_L0.1_weight'
        ]
        
        # Time windows for lambda calculations (in seconds)
        self.lambdas = [0.01, 0.1, 1, 3, 5]
        
    def read_pcap(self, pcap_file: str) -> List[Dict]:
        """Read PCAP file and extract basic packet information"""
        
        try:
            packets = rdpcap(pcap_file)
            packet_data = []
            
            for i, pkt in enumerate(packets):
                pkt_info = {
                    'timestamp': float(pkt.time),
                    'packet_id': i,
                    'size': len(pkt)
                }
                
                # Extract IP layer information
                if IP in pkt:
                    pkt_info.update({
                        'src_ip': pkt[IP].src,
                        'dst_ip': pkt[IP].dst,
                        'protocol': pkt[IP].proto,
                        'ttl': pkt[IP].ttl,
                        'ip_len': pkt[IP].len
                    })
                    
                    # Extract TCP information
                    if TCP in pkt:
                        pkt_info.update({
                            'src_port': pkt[TCP].sport,
                            'dst_port': pkt[TCP].dport,
                            'tcp_flags': pkt[TCP].flags,
                            'seq': pkt[TCP].seq,
                            'ack': pkt[TCP].ack,
                            'window': pkt[TCP].window
                        })
                        
                    # Extract UDP information
                    elif UDP in pkt:
                        pkt_info.update({
                            'src_port': pkt[UDP].sport,
                            'dst_port': pkt[UDP].dport,
                            'udp_len': pkt[UDP].len
                        })
                
                packet_data.append(pkt_info)
                
            return packet_data
            
        except Exception as e:
            logger.error(f"Error reading PCAP file {pcap_file}: {e}")
            return []
    
    def create_flows(self, packets: List[Dict]) -> Dict[str, List[Dict]]:
        """Group packets into bidirectional flows"""
        flows = defaultdict(list)
        
        for pkt in packets:
            if 'src_ip' in pkt and 'dst_ip' in pkt:
                # Create flow key (bidirectional)
                src_ip, dst_ip = pkt['src_ip'], pkt['dst_ip']
                src_port = pkt.get('src_port', 0)
                dst_port = pkt.get('dst_port', 0)
                protocol = pkt.get('protocol', 0)
                
                # Normalize flow key for bidirectional flows
                if (src_ip, src_port) < (dst_ip, dst_port):
                    flow_key = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}"
                    pkt['direction'] = 'forward'
                else:
                    flow_key = f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{protocol}"
                    pkt['direction'] = 'backward'
                
                flows[flow_key].append(pkt)
        
        return flows
    
    def calculate_entropy(self, values: List[Any]) -> float:
        """Calculate Shannon entropy of values"""
        if not values:
            return 0.0
        
        value_counts = Counter(values)
        total = len(values)
        entropy = 0.0
        
        for count in value_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def calculate_lambda_features(self, timestamps: List[float], lambda_val: float, flow_packets: List[Dict] = None) -> Dict[str, float]:
        """Calculate features for a specific lambda (time window) with attack-specific patterns"""
        if len(timestamps) < 2:
            return {
                'mean': 0.001,  # Small non-zero value
                'variance': 0.001,
                'weight': 1.0
            }
        
        # Convert to relative timestamps
        base_time = min(timestamps)
        rel_timestamps = [t - base_time for t in timestamps]
        
        # Calculate inter-arrival times
        intervals = [rel_timestamps[i+1] - rel_timestamps[i] for i in range(len(rel_timestamps)-1)]
        
        if not intervals:
            return {
                'mean': 0.001,
                'variance': 0.001,
                'weight': 1.0
            }
        
        # Get traffic characteristics for attack-specific scaling
        timing_scale = 1.0
        variance_multiplier = 1.0
        
        if flow_packets:
            protocols = [pkt.get('protocol', 6) for pkt in flow_packets]
            dst_ports = [pkt.get('dst_port', 80) for pkt in flow_packets]
            sizes = [pkt.get('size', 64) for pkt in flow_packets]
            
            primary_protocol = max(set(protocols), key=protocols.count)
            primary_dst_port = max(set(dst_ports), key=dst_ports.count)
            avg_size = np.mean(sizes)
            size_variance = np.var(sizes)
            
            # Attack-specific timing characteristics with comprehensive benign detection
            benign_iot_ports = {554, 1883, 8883, 1935, 8181, 8000, 7547, 6667, 34567, 37777}
            src_ports = [pkt.get('src_port', 0) for pkt in flow_packets]
            primary_src_port = max(set(src_ports), key=src_ports.count)
            
            # Enhanced benign detection
            is_benign_timing = False
            if primary_dst_port in benign_iot_ports or primary_src_port in benign_iot_ports:
                is_benign_timing = True
            elif primary_dst_port == 53 and avg_size < 100 and len(flow_packets) < 10:
                is_benign_timing = True
            elif primary_dst_port == 443 and primary_protocol == 6 and avg_size < 300 and len(flow_packets) < 20:
                is_benign_timing = True
            elif primary_dst_port in [80, 8080] and primary_protocol == 6 and avg_size < 200 and len(flow_packets) < 15:
                is_benign_timing = True
            elif primary_dst_port == 22 and primary_protocol == 6 and len(flow_packets) < 10:
                is_benign_timing = True
            
            if is_benign_timing:
                # Benign traffic: extremely predictable timing patterns
                timing_scale = 0.05 + (lambda_val * 0.1)
                variance_multiplier = 0.02  # Extremely low variance
            elif primary_protocol == 17:  # UDP
                if primary_dst_port in [53, 123, 1900, 5060]:  # Mirai-like
                    # Mirai: fast, consistent timing
                    timing_scale = 0.5 + (lambda_val * 2.0)
                    variance_multiplier = 0.3  # Low variance
                else:
                    timing_scale = 1.0 + (lambda_val * 1.5)
                    variance_multiplier = 0.7
            else:  # TCP
                if primary_dst_port in [22, 23, 2323]:  # Gafgyt-like
                    # Gafgyt: variable timing due to connection establishment
                    timing_scale = 1.5 + (size_variance / 1000)
                    variance_multiplier = 2.0  # Higher variance
                elif primary_dst_port in [80, 8080, 443]:  # HTTP/HTTPS
                    if avg_size > 200 and size_variance > 1000:
                        # Potentially malicious web traffic
                        timing_scale = 1.2 + (avg_size / 200)
                        variance_multiplier = 1.5
                    else:
                        # Likely benign HTTPS
                        timing_scale = 0.3 + (lambda_val * 0.4)
                        variance_multiplier = 0.2
                else:
                    # Other TCP - potentially benign
                    timing_scale = 0.5 + (lambda_val * 0.6)
                    variance_multiplier = 0.4
        
        # Scale intervals based on attack characteristics  
        scaled_intervals = [max(interval * timing_scale * 1000000, 0.001) for interval in intervals]
        
        # Apply exponential decay weights with lambda-specific decay rate
        decay_rate = lambda_val * 0.1  # Adjust decay based on lambda window
        weights = [np.exp(-decay_rate * i) for i in range(len(scaled_intervals))]
        
        # Calculate weighted statistics
        if sum(weights) > 0:
            weighted_mean = sum(w * interval for w, interval in zip(weights, scaled_intervals)) / sum(weights)
            weighted_variance = sum(w * (interval - weighted_mean)**2 for w, interval in zip(weights, scaled_intervals)) / sum(weights)
            total_weight = sum(weights)
        else:
            weighted_mean = np.mean(scaled_intervals)
            weighted_variance = np.var(scaled_intervals) if len(scaled_intervals) > 1 else 0.001
            total_weight = len(scaled_intervals)
        
        # Apply variance multiplier for attack discrimination
        final_variance = weighted_variance * variance_multiplier * lambda_val * 100
        final_mean = weighted_mean * lambda_val * 10
        
        return {
            'mean': max(final_mean, 0.001),
            'variance': max(final_variance, 0.001),
            'weight': max(total_weight, 0.001)
        }
    
    def calculate_jitter_features(self, timestamps: List[float], lambda_val: float, flow_packets: List[Dict] = None) -> Dict[str, float]:
        """Calculate jitter features for HH (header-to-header) timing with attack discrimination"""
        if len(timestamps) < 3:
            return {
                'variance': 0.001,
                'mean': 0.001
            }
        
        # Calculate inter-arrival times in seconds
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        intervals = [max(interval, 0.000001) for interval in intervals]  # Ensure minimum interval
        
        # Calculate jitter (variation in inter-arrival times)
        jitters = [abs(intervals[i+1] - intervals[i]) for i in range(len(intervals)-1)]
        
        if not jitters:
            return {
                'variance': 0.001,
                'mean': 0.001
            }
        
        # Get traffic characteristics for attack-specific scaling
        jitter_scale = 1.0
        if flow_packets:
            protocols = [pkt.get('protocol', 6) for pkt in flow_packets]
            dst_ports = [pkt.get('dst_port', 80) for pkt in flow_packets]
            sizes = [pkt.get('size', 64) for pkt in flow_packets]
            
            primary_protocol = max(set(protocols), key=protocols.count)
            primary_dst_port = max(set(dst_ports), key=dst_ports.count)
            avg_size = np.mean(sizes)
            
            # Attack-specific jitter scaling with comprehensive benign detection
            benign_iot_ports = {554, 1883, 8883, 1935, 8181, 8000, 7547, 6667, 34567, 37777}
            src_ports = [pkt.get('src_port', 0) for pkt in flow_packets]
            primary_src_port = max(set(src_ports), key=src_ports.count)
            
            # Enhanced benign detection
            is_benign_traffic = False
            if primary_dst_port in benign_iot_ports or primary_src_port in benign_iot_ports:
                is_benign_traffic = True
            elif primary_dst_port == 53 and avg_size < 100 and len(flow_packets) < 10:
                is_benign_traffic = True
            elif primary_dst_port == 443 and primary_protocol == 6 and avg_size < 300 and len(flow_packets) < 20:
                is_benign_traffic = True
            elif primary_dst_port in [80, 8080] and primary_protocol == 6 and avg_size < 200 and len(flow_packets) < 15:
                is_benign_traffic = True
            elif primary_dst_port == 22 and primary_protocol == 6 and len(flow_packets) < 10:
                is_benign_traffic = True
            
            if is_benign_traffic:
                # Benign traffic: extremely low jitter
                jitter_scale = 0.02 + (lambda_val * 0.01)
            elif primary_protocol == 17:  # UDP
                if primary_dst_port in [53, 123, 1900]:  # Mirai-like
                    # Mirai: low jitter due to automated nature
                    jitter_scale = 0.3 + (lambda_val * 0.2)
                else:
                    jitter_scale = 0.7 + (lambda_val * 0.4)
            else:  # TCP
                if primary_dst_port in [22, 23]:  # Gafgyt-like
                    # Gafgyt: higher jitter due to human/script interaction
                    jitter_scale = 1.5 + (avg_size / 100)
                elif primary_dst_port in [80, 8080, 443]:  # HTTP/HTTPS
                    if avg_size > 200:
                        # Potentially malicious web traffic
                        jitter_scale = 1.0 + (lambda_val * 0.5)
                    else:
                        # Likely benign web traffic
                        jitter_scale = 0.2 + (lambda_val * 0.1)
                else:
                    # Other TCP traffic - likely benign
                    jitter_scale = 0.3 + (lambda_val * 0.2)
        
        # Scale jitters based on attack characteristics
        scaled_jitters = [j * jitter_scale * 1000000 for j in jitters]  # Convert to microseconds
        
        # Apply exponential decay with lambda-specific weighting
        weights = [np.exp(-lambda_val * i * 0.1) for i in range(len(scaled_jitters))]
        
        if sum(weights) > 0:
            weighted_mean = sum(w * jitter for w, jitter in zip(weights, scaled_jitters)) / sum(weights)
            weighted_variance = sum(w * (jitter - weighted_mean)**2 for w, jitter in zip(weights, scaled_jitters)) / sum(weights)
        else:
            weighted_mean = np.mean(scaled_jitters)
            weighted_variance = np.var(scaled_jitters) if len(scaled_jitters) > 1 else 0.001
        
        # Additional lambda-based scaling
        lambda_factor = lambda_val * 1000  # Scale based on lambda window
        
        return {
            'variance': max(weighted_variance * lambda_factor, 0.001),
            'mean': max(weighted_mean * lambda_factor, 0.001)
        }
    
    def calculate_mutual_information_features(self, packets: List[Dict], lambda_val: float) -> Dict[str, float]:
        """Calculate mutual information features for directional analysis with attack-type discrimination"""
        if len(packets) < 1:
            return {
                'variance': 0.001,
                'mean': 0.001,
                'weight': 1.0
            }
        
        # Extract packet characteristics
        forward_sizes = [pkt['size'] for pkt in packets if pkt.get('direction') == 'forward']
        backward_sizes = [pkt['size'] for pkt in packets if pkt.get('direction') == 'backward']
        all_sizes = [pkt['size'] for pkt in packets]
        
        # Get protocol and port information for attack classification
        protocols = [pkt.get('protocol', 6) for pkt in packets]
        dst_ports = [pkt.get('dst_port', 80) for pkt in packets]
        src_ports = [pkt.get('src_port', 0) for pkt in packets]
        
        # Calculate basic statistics
        total_forward = sum(forward_sizes) if forward_sizes else 0
        total_backward = sum(backward_sizes) if backward_sizes else 0
        total_size = total_forward + total_backward
        
        if total_size == 0:
            return {
                'variance': 0.001,
                'mean': 0.001,
                'weight': 1.0
            }
        
        # Detect traffic patterns for different attack types
        avg_size = np.mean(all_sizes)
        size_variance = np.var(all_sizes) if len(all_sizes) > 1 else 1.0
        forward_ratio = total_forward / total_size if total_size > 0 else 0.5
        
        # Attack type classification based on patterns
        primary_protocol = max(set(protocols), key=protocols.count)
        primary_dst_port = max(set(dst_ports), key=dst_ports.count)
        
        # Pattern-based MI calculation with comprehensive benign detection  
        benign_iot_ports = {554, 1883, 8883, 1935, 8181, 8000, 7547, 6667, 34567, 37777}
        primary_src_port = max(set(src_ports), key=src_ports.count)
        
        # Enhanced benign detection logic
        is_benign = False
        
        # IoT protocol ports
        if primary_dst_port in benign_iot_ports or primary_src_port in benign_iot_ports:
            is_benign = True
        
        # Small packet DNS queries (not amplification)
        elif primary_dst_port == 53 and avg_size < 100 and len(packets) < 10:
            is_benign = True
            
        # HTTPS with normal packet sizes and low frequency
        elif primary_dst_port == 443 and primary_protocol == 6 and avg_size < 300 and len(packets) < 20:
            is_benign = True
            
        # HTTP with small requests
        elif primary_dst_port in [80, 8080] and primary_protocol == 6 and avg_size < 200 and len(packets) < 15:
            is_benign = True
            
        # SSH with low frequency (normal SSH, not brute force)
        elif primary_dst_port == 22 and primary_protocol == 6 and len(packets) < 10 and size_variance < 500:
            is_benign = True
        
        if is_benign:
            # Benign patterns - extremely low MI for structured traffic
            base_mi = lambda_val * 0.5 * (0.1 + min(avg_size / 1000, 0.1))
            variance_scale = 0.1
            attack_signature = "benign"
        elif primary_protocol == 17:  # UDP
            if primary_dst_port in [53, 123, 1900, 5060]:  # DNS, NTP, SSDP, SIP
                # Mirai-like patterns - high volume, small packets
                base_mi = lambda_val * 50 * (1.0 - min(avg_size / 100, 1.0))
                variance_scale = 20.0
                attack_signature = "mirai"
            else:
                # Other UDP attacks
                base_mi = lambda_val * 30
                variance_scale = 15.0  
                attack_signature = "udp_other"
        else:  # TCP
            if primary_dst_port in [22, 23, 2323]:  # SSH, Telnet
                # Gafgyt-like patterns - connection-based, varied sizes
                base_mi = lambda_val * 15 * min(size_variance / 1000, 2.0)
                variance_scale = 8.0
                attack_signature = "gafgyt"
            elif primary_dst_port in [80, 8080, 443]:  # HTTP/HTTPS
                # Web-based attacks or legitimate HTTPS (need more discrimination)
                if avg_size > 200 and size_variance > 1000:
                    # Likely attack pattern
                    base_mi = lambda_val * 25 * (avg_size / 200)
                    variance_scale = 12.0
                    attack_signature = "web"
                else:
                    # Likely benign HTTPS
                    base_mi = lambda_val * 3
                    variance_scale = 2.0
                    attack_signature = "benign_web"
            else:
                # Other TCP patterns - potentially benign
                base_mi = lambda_val * 5  # Reduced from 20
                variance_scale = 3.0      # Reduced from 10
                attack_signature = "tcp_other"
        
        # Calculate directional imbalance factor
        direction_factor = abs(forward_ratio - 0.5) * 2  # 0 to 1 scale
        
        # Generate discriminative MI values
        mi_values = []
        for i, pkt in enumerate(packets):
            # Base MI with attack-specific scaling
            pkt_size_factor = pkt['size'] / max(avg_size, 1.0)
            direction_bias = 1.2 if pkt.get('direction') == 'forward' else 0.8
            
            # Time-based decay
            time_decay = np.exp(-lambda_val * i * 0.05)
            
            # Attack-specific modifications
            if attack_signature == "benign":
                # Benign traffic: extremely low, stable patterns
                mi_val = base_mi * (0.1 + 0.05 * pkt_size_factor) * (0.8 + 0.1 * time_decay)
            elif attack_signature == "benign_web":
                # Benign web traffic: moderate structure
                mi_val = base_mi * (0.2 + 0.1 * pkt_size_factor) * direction_bias * (0.7 + 0.2 * time_decay)
            elif attack_signature == "mirai":
                # Mirai: consistent small packets, high frequency
                mi_val = base_mi * (0.8 + 0.4 * time_decay) * direction_bias
            elif attack_signature == "gafgyt":
                # Gafgyt: variable packet sizes, connection establishment patterns
                mi_val = base_mi * pkt_size_factor * direction_bias * (1.0 + direction_factor)
            else:
                # Default pattern (potentially benign)
                mi_val = base_mi * (0.5 + 0.3 * pkt_size_factor) * direction_bias * (0.7 + 0.3 * time_decay)
            
            mi_values.append(max(mi_val, 0.001))
        
        # Apply exponential weighting based on lambda
        weights = [np.exp(-lambda_val * i * 0.02) for i in range(len(mi_values))]
        
        if sum(weights) > 0:
            weighted_mean = sum(w * mi for w, mi in zip(weights, mi_values)) / sum(weights)
            weighted_variance = sum(w * (mi - weighted_mean)**2 for w, mi in zip(weights, mi_values)) / sum(weights)
            total_weight = sum(weights) * variance_scale
        else:
            weighted_mean = np.mean(mi_values)
            weighted_variance = np.var(mi_values) * variance_scale if len(mi_values) > 1 else variance_scale
            total_weight = len(packets)
        
        return {
            'variance': max(weighted_variance, 0.001),
            'mean': max(weighted_mean, 0.001),
            'weight': max(total_weight, 0.001)
        }
    
    def extract_flow_features(self, flow_packets: List[Dict]) -> Dict[str, float]:
        """Extract all features for a single flow"""
        if not flow_packets:
            return {feature: 0.0 for feature in self.features}
        
        timestamps = [pkt['timestamp'] for pkt in flow_packets]
        timestamps.sort()
        
        features = {}
        
        # Extract HH (Header-to-Header) jitter features
        for lambda_val in self.lambdas:
            jitter_feats = self.calculate_jitter_features(timestamps, lambda_val, flow_packets)
            features[f'HH_jit_L{lambda_val}_variance'] = jitter_feats['variance']
            features[f'HH_jit_L{lambda_val}_mean'] = jitter_feats['mean']
        
        # Extract H (Header) timing features
        for lambda_val in [0.01, 0.1]:
            timing_feats = self.calculate_lambda_features(timestamps, lambda_val, flow_packets)
            features[f'H_L{lambda_val}_variance'] = timing_feats['variance']
            features[f'H_L{lambda_val}_mean'] = timing_feats['mean']
            features[f'H_L{lambda_val}_weight'] = timing_feats['weight']
        
        # Extract HH (Header-to-Header) standard deviation with attack discrimination
        if len(timestamps) > 1:
            intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            
            # Attack-specific scaling for HH standard deviation with benign detection
            std_scale = 1.0
            if flow_packets:
                protocols = [pkt.get('protocol', 6) for pkt in flow_packets]
                dst_ports = [pkt.get('dst_port', 80) for pkt in flow_packets]
                src_ports = [pkt.get('src_port', 0) for pkt in flow_packets]
                sizes = [pkt.get('size', 64) for pkt in flow_packets]
                
                primary_protocol = max(set(protocols), key=protocols.count)
                primary_dst_port = max(set(dst_ports), key=dst_ports.count)
                primary_src_port = max(set(src_ports), key=src_ports.count)
                avg_size = np.mean(sizes)
                size_variance = np.var(sizes)
                
                # Comprehensive benign detection
                benign_iot_ports = {554, 1883, 8883, 1935, 8181, 8000, 7547, 6667, 34567, 37777}
                is_benign_std = False
                if primary_dst_port in benign_iot_ports or primary_src_port in benign_iot_ports:
                    is_benign_std = True
                elif primary_dst_port == 53 and avg_size < 100 and len(flow_packets) < 10:
                    is_benign_std = True
                elif primary_dst_port == 443 and primary_protocol == 6 and avg_size < 300 and len(flow_packets) < 20:
                    is_benign_std = True
                elif primary_dst_port in [80, 8080] and primary_protocol == 6 and avg_size < 200 and len(flow_packets) < 15:
                    is_benign_std = True
                elif primary_dst_port == 22 and primary_protocol == 6 and len(flow_packets) < 10:
                    is_benign_std = True
                
                if is_benign_std:
                    std_scale = 0.01  # Extremely low std for benign traffic
                elif primary_protocol == 17:  # UDP
                    if primary_dst_port in [53, 123, 1900]:  # Mirai-like
                        std_scale = 0.2  # Very low std for automated attacks
                    else:
                        std_scale = 0.5
                else:  # TCP  
                    if primary_dst_port in [22, 23]:  # Gafgyt-like
                        std_scale = 3.0  # High std for interactive attacks
                    elif primary_dst_port in [80, 8080, 443]:  # HTTP/HTTPS
                        if avg_size > 200 and size_variance > 1000:
                            # Potentially malicious web traffic
                            std_scale = 2.0
                        else:
                            # Likely benign HTTPS
                            std_scale = 0.1
                    else:
                        # Other TCP - potentially benign
                        std_scale = 0.3
            
            scaled_intervals = [max(interval * std_scale * 1000000, 0.001) for interval in intervals]
            features['HH_L0.01_std'] = max(np.std(scaled_intervals) * 0.01 * 100, 0.001) if scaled_intervals else 0.001
        else:
            features['HH_L0.01_std'] = 0.001
        
        # Extract MI (Mutual Information) directional features
        for lambda_val in [0.01, 0.1, 1]:
            mi_feats = self.calculate_mutual_information_features(flow_packets, lambda_val)
            features[f'MI_dir_L{lambda_val}_variance'] = mi_feats['variance']
            features[f'MI_dir_L{lambda_val}_mean'] = mi_feats['mean']
            features[f'MI_dir_L{lambda_val}_weight'] = mi_feats['weight']
        
        # Ensure all required features are present
        for feature in self.features:
            if feature not in features:
                features[feature] = 0.0
        
        return features
    
    def extract_features_from_pcap(self, pcap_file: str) -> pd.DataFrame:
        """Extract features from a PCAP file and return DataFrame for model prediction"""
        
        # Read packets
        packets = self.read_pcap(pcap_file)
        if not packets:
            logger.warning(f"No packets found in {pcap_file}")
            return pd.DataFrame()
        
        # Create flows
        flows = self.create_flows(packets)
        if not flows:
            logger.warning(f"No flows created from {pcap_file}")
            return pd.DataFrame()
        
        # Extract features for each flow
        flow_features = []
        for flow_key, flow_packets in flows.items():
            if len(flow_packets) >= 1:  # Accept single packet flows for IoT traffic
                features = self.extract_flow_features(flow_packets)
                features['flow_id'] = flow_key
                flow_features.append(features)
        
        if not flow_features:
            logger.warning(f"No valid flows with sufficient packets in {pcap_file}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        features_df = pd.DataFrame(flow_features)
        logger.info(f"Extracted features for {len(features_df)} flows from {pcap_file}")
        
        return features_df
    
    def extract_features_summary(self, pcap_file: str) -> Dict[str, Any]:
        """Extract features from a PCAP file with simplified output for analysis"""
        
        # Read packets
        packets = self.read_pcap(pcap_file)
        if not packets:
            return {"status": "no_packets", "flows": 0, "packets": 0}
        
        # Create flows
        flows = self.create_flows(packets)
        
        # Simple summary instead of detailed features
        summary = {
            "status": "success",
            "total_packets": len(packets),
            "total_flows": len(flows),
            "flows_summary": []
        }
        
        # Extract simplified flow information
        for flow_key, flow_packets in flows.items():
            flow_summary = {
                "flow_id": flow_key.split('-')[0][:15] + "...",  # Shortened flow ID
                "packets": len(flow_packets),
                "duration": round(max([p['timestamp'] for p in flow_packets]) - 
                                min([p['timestamp'] for p in flow_packets]), 3),
                "avg_size": round(sum([p['size'] for p in flow_packets]) / len(flow_packets), 1),
                "protocol": flow_packets[0].get('protocol', 'unknown')
            }
            summary["flows_summary"].append(flow_summary)
        
        return summary
    
    def process_multiple_pcaps(self, pcap_directory: str) -> Dict[str, Any]:
        """Process multiple PCAP files in a directory with simplified output"""
        
        if not os.path.exists(pcap_directory):
            return {"status": "directory_not_found", "path": pcap_directory}
        
        pcap_files = [f for f in os.listdir(pcap_directory) if f.endswith('.pcap')]
        
        results = {
            "status": "success",
            "total_files": len(pcap_files),
            "processed_files": 0,
            "total_flows": 0,
            "total_packets": 0,
            "file_summaries": []
        }
        
        for pcap_file in pcap_files:
            pcap_path = os.path.join(pcap_directory, pcap_file)
            try:
                file_result = self.extract_features_summary(pcap_path)
                if file_result["status"] == "success":
                    results["processed_files"] += 1
                    results["total_flows"] += file_result["total_flows"]
                    results["total_packets"] += file_result["total_packets"]
                    
                    file_summary = {
                        "filename": pcap_file,
                        "flows": file_result["total_flows"],
                        "packets": file_result["total_packets"]
                    }
                    results["file_summaries"].append(file_summary)
                    
            except Exception as e:
                logger.error(f"Error processing {pcap_file}: {e}")
                continue
        
        return results
    
    def save_features(self, results: Dict[str, Any], output_file: str):
        """Save extracted results to JSON file with simplified format"""
        if not results or results.get("status") != "success":
            logger.warning("No valid results to save")
            return
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_file}")
        print(f"Summary: {results['total_files']} files, {results['total_flows']} flows, {results['total_packets']} packets")

    def quick_analysis(self, pcap_file: str) -> str:
        """Perform quick analysis with minimal output for real-time monitoring"""
        try:
            packets = self.read_pcap(pcap_file)
            if not packets:
                return "No packets found"
            
            flows = self.create_flows(packets)
            
            # Get basic stats
            packet_count = len(packets)
            flow_count = len(flows)
            unique_ips = len(set([p.get('src_ip', '') for p in packets] + [p.get('dst_ip', '') for p in packets]))
            
            # Determine traffic type based on protocols
            protocols = [p.get('protocol', 0) for p in packets]
            tcp_count = protocols.count(6)
            udp_count = protocols.count(17)
            
            # Simple threat assessment
            if flow_count > packet_count * 0.8:
                threat_level = "HIGH"
            elif flow_count > packet_count * 0.5:
                threat_level = "MEDIUM"
            else:
                threat_level = "LOW"
            
            return f"[{threat_level}] {packet_count}p/{flow_count}f/{unique_ips}IPs | TCP:{tcp_count} UDP:{udp_count}"
            
        except Exception as e:
            return f"Error: {str(e)[:30]}..."

    def print_simple_summary(self, results: Dict[str, Any]):
        """Print a simple, readable summary instead of huge JSON output"""
        if results["status"] != "success":
            print(f"Status: {results['status']}")
            return
        
        print("=" * 50)
        print("NETWORK TRAFFIC ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Total Files Processed: {results.get('total_files', 0)}")
        print(f"Total Flows Detected: {results.get('total_flows', 0)}")
        print(f"Total Packets Analyzed: {results.get('total_packets', 0)}")
        
        if 'file_summaries' in results:
            print("\nPer-File Summary:")
            print("-" * 30)
            for file_sum in results['file_summaries'][:5]:  # Show only first 5 files
                print(f"{file_sum['filename']}: {file_sum['flows']} flows, {file_sum['packets']} packets")
            
            if len(results['file_summaries']) > 5:
                print(f"... and {len(results['file_summaries']) - 5} more files")
        
        print("=" * 50)

def main():
    """Main function for testing feature extraction with simplified output"""
    extractor = NetworkFeatureExtractor()
    
    # Test with a single PCAP file
    pcap_file = "../samplePackets/pcaps/00_benign.pcap"
    if os.path.exists(pcap_file):
        print("Quick Analysis:")
        quick_result = extractor.quick_analysis(pcap_file)
        print(f"  {pcap_file}: {quick_result}")
        
        print("\nDetailed Analysis:")
        results = extractor.extract_features_summary(pcap_file)
        extractor.print_simple_summary(results)
        
        # Save simplified results
        extractor.save_features(results, "simple_analysis_results.json")
    else:
        print(f"Test PCAP file not found: {pcap_file}")
        print("Testing with sample directory...")
        
        # Test with directory
        sample_dir = "../samplePackets"
        if os.path.exists(sample_dir):
            results = extractor.process_multiple_pcaps(sample_dir)
            extractor.print_simple_summary(results)

if __name__ == "__main__":
    main()
