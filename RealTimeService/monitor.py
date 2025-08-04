#!/usr/bin/env python3
"""
Real-time Monitor for IoT Threat Detection
Continuously monitors a directory for new PCAP files and processes them automatically
"""

import os
import time
import logging
import json
from datetime import datetime
from typing import Set
import argparse
from pathlib import Path

from realtime_pipeline import RealTimeThreatDetectionPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealTimeMonitor:
    """
    Monitors a directory for new PCAP files and processes them automatically
    """
    
    def __init__(self, pcap_path: str, model_path: str, poll_interval: int = 5):
        self.pcap_path = pcap_path
        self.model_path = model_path
        self.poll_interval = poll_interval
        self.processed_files: Set[str] = set()
        
        # Initialize pipeline
        self.pipeline = RealTimeThreatDetectionPipeline(pcap_path, model_path)
        
        # Create monitoring log
        self.monitor_log = "monitor.log"
        self.stats_file = "monitoring_stats.json"
        
        logger.info(f"Monitor initialized - polling every {poll_interval} seconds")
    
    def get_pcap_files(self) -> Set[str]:
        """Get all PCAP files in the monitored directory"""
        if not os.path.exists(self.pcap_path):
            return set()
        
        pcap_files = {
            os.path.join(self.pcap_path, f) 
            for f in os.listdir(self.pcap_path) 
            if f.endswith('.pcap')
        }
        return pcap_files
    
    def process_new_file(self, pcap_file: str):
        """Process a newly detected PCAP file"""
        logger.info(f"New file detected: {pcap_file}")
        
        try:
            # Process the file
            result = self.pipeline.process_single_pcap(pcap_file)
            
            # Log the result
            self.log_result(pcap_file, result)
            
            # Update statistics
            self.update_stats(result)
            
            # Mark as processed
            self.processed_files.add(pcap_file)
            
            logger.info(f"Successfully processed: {pcap_file}")
            
        except Exception as e:
            logger.error(f"Error processing {pcap_file}: {e}")
    
    def log_result(self, pcap_file: str, result: dict):
        """Log processing result"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'file': pcap_file,
            'status': result.get('status', 'unknown'),
            'flows': result.get('total_flows', 0),
            'malicious_flows': result.get('analysis', {}).get('malicious_flows', 0),
            'processing_time': result.get('processing_time', 0)
        }
        
        # Write to monitoring log
        with open(self.monitor_log, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def update_stats(self, result: dict):
        """Update monitoring statistics"""
        # Load existing stats
        stats = self.load_stats()
        
        # Update counters
        stats['total_files_processed'] += 1
        
        if result.get('status') == 'success':
            stats['successful_processes'] += 1
            analysis = result.get('analysis', {})
            stats['total_flows_analyzed'] += analysis.get('total_flows', 0)
            stats['total_malicious_detected'] += analysis.get('malicious_flows', 0)
            stats['total_processing_time'] += result.get('processing_time', 0)
        else:
            stats['failed_processes'] += 1
        
        stats['last_updated'] = datetime.now().isoformat()
        
        # Save updated stats
        self.save_stats(stats)
    
    def load_stats(self) -> dict:
        """Load monitoring statistics"""
        default_stats = {
            'monitoring_started': datetime.now().isoformat(),
            'total_files_processed': 0,
            'successful_processes': 0,
            'failed_processes': 0,
            'total_flows_analyzed': 0,
            'total_malicious_detected': 0,
            'total_processing_time': 0.0,
            'last_updated': datetime.now().isoformat()
        }
        
        try:
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load stats: {e}")
        
        return default_stats
    
    def save_stats(self, stats: dict):
        """Save monitoring statistics"""
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save stats: {e}")
    
    def print_stats(self):
        """Print current monitoring statistics"""
        stats = self.load_stats()
        
        print("\n" + "="*50)
        print("REAL-TIME MONITORING STATISTICS")
        print("="*50)
        print(f"Monitoring started: {stats['monitoring_started']}")
        print(f"Last updated: {stats['last_updated']}")
        print(f"Total files processed: {stats['total_files_processed']}")
        print(f"Successful: {stats['successful_processes']}")
        print(f"Failed: {stats['failed_processes']}")
        
        if stats['successful_processes'] > 0:
            success_rate = (stats['successful_processes'] / stats['total_files_processed']) * 100
            avg_time = stats['total_processing_time'] / stats['successful_processes']
            print(f"Success rate: {success_rate:.1f}%")
            print(f"Average processing time: {avg_time:.2f} seconds")
            print(f"Total flows analyzed: {stats['total_flows_analyzed']}")
            print(f"Total malicious flows detected: {stats['total_malicious_detected']}")
            
            if stats['total_flows_analyzed'] > 0:
                malicious_rate = (stats['total_malicious_detected'] / stats['total_flows_analyzed']) * 100
                print(f"Malicious flow rate: {malicious_rate:.1f}%")
        
        print("="*50)
    
    def run(self):
        """Start monitoring loop"""
        logger.info(f"Starting real-time monitoring of: {self.pcap_path}")
        logger.info(f"Using model: {self.model_path}")
        logger.info(f"Poll interval: {self.poll_interval} seconds")
        
        # Initialize processed files with existing files
        self.processed_files = self.get_pcap_files()
        logger.info(f"Found {len(self.processed_files)} existing files - will monitor for new files")
        
        try:
            while True:
                # Get current files
                current_files = self.get_pcap_files()
                
                # Find new files
                new_files = current_files - self.processed_files
                
                if new_files:
                    logger.info(f"Found {len(new_files)} new files")
                    for new_file in new_files:
                        self.process_new_file(new_file)
                
                # Print stats periodically
                if len(self.processed_files) % 10 == 0 and len(self.processed_files) > 0:
                    self.print_stats()
                
                # Wait for next poll
                time.sleep(self.poll_interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
            self.print_stats()
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            self.print_stats()

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Real-time IoT Threat Detection Monitor')
    parser.add_argument(
        '--pcap-path',
        default='../samplePackets/pcaps',
        help='Path to monitor for PCAP files (default: ../samplePackets/pcaps)'
    )
    parser.add_argument(
        '--model-path',
        default='../SavedGlobalModel/final_model.pth',
        help='Path to trained model file (default: ../SavedGlobalModel/final_model.pth)'
    )
    parser.add_argument(
        '--poll-interval',
        type=int,
        default=5,
        help='Polling interval in seconds (default: 5)'
    )
    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Only print current statistics and exit'
    )
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = RealTimeMonitor(args.pcap_path, args.model_path, args.poll_interval)
    
    if args.stats_only:
        monitor.print_stats()
    else:
        monitor.run()

if __name__ == "__main__":
    main()
