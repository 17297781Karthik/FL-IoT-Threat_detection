import smtplib
import json
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

class ThreatEmailAlerter:
    
    def __init__(self):
        load_dotenv()
        
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.email_username = os.getenv('EMAIL_USERNAME')
        self.email_password = os.getenv('EMAIL_PASSWORD')
        self.email_from = os.getenv('EMAIL_FROM')
        self.email_to = os.getenv('EMAIL_TO')
        
        if not all([self.email_username, self.email_password, self.email_from, self.email_to]):
            raise ValueError("Email credentials not properly configured in .env file")
        
        self.resources_path = os.path.join(os.path.dirname(__file__), 'resources')
        self.attack_info = self._load_attack_info()
    
    def _load_attack_info(self) -> Dict[str, Any]:
        attack_info = {}
        for attack_type in ['gafgyt', 'mirai']:
            file_path = os.path.join(self.resources_path, f'{attack_type}.json')
            try:
                with open(file_path, 'r') as f:
                    attack_info[attack_type] = json.load(f)
            except Exception as e:
                logger.error(f"Error loading attack info for {attack_type}: {e}")
        return attack_info
    
    def _get_attack_category(self, attack_name: str) -> str:
        attack_lower = attack_name.lower()
        if 'gafgyt' in attack_lower:
            return 'gafgyt'
        elif 'mirai' in attack_lower:
            return 'mirai'
        return None
    
    def _create_email_body(self, detection_data: Dict[str, Any]) -> str:
        attack_type = detection_data.get('attack_type', 'Unknown')
        category = self._get_attack_category(attack_type)
        
        if not category or category not in self.attack_info:
            return self._create_basic_alert(detection_data)
        
        info = self.attack_info[category]
        
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 800px; margin: 0 auto; padding: 20px;">
                <div style="background-color: #d32f2f; color: white; padding: 20px; border-radius: 5px 5px 0 0;">
                    <h1 style="margin: 0;">IoT Threat Detection Alert</h1>
                    <p style="margin: 5px 0 0 0; font-size: 14px;">Detected: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div style="background-color: #f5f5f5; padding: 20px; border-left: 4px solid #d32f2f;">
                    <h2 style="color: #d32f2f; margin-top: 0;">Threat Detected: {info['name']}</h2>
                    <p><strong>Severity Level:</strong> <span style="color: #d32f2f; font-weight: bold;">{info['severity']}</span></p>
                    <p><strong>Attack Variant:</strong> {attack_type}</p>
                    <p><strong>Confidence Level:</strong> {detection_data.get('confidence', 'N/A')}</p>
                </div>
                
                <div style="background-color: white; padding: 20px; border: 1px solid #ddd; margin-top: 20px;">
                    <h3 style="color: #1976d2; border-bottom: 2px solid #1976d2; padding-bottom: 10px;">Threat Description</h3>
                    <p style="text-align: justify;">{info['description']}</p>
                </div>
                
                <div style="background-color: white; padding: 20px; border: 1px solid #ddd; margin-top: 20px;">
                    <h3 style="color: #1976d2; border-bottom: 2px solid #1976d2; padding-bottom: 10px;">Attack Characteristics</h3>
                    <ul>
        """
        
        for characteristic in info['characteristics']:
            html += f"                        <li>{characteristic}</li>\n"
        
        html += f"""
                    </ul>
                </div>
                
                <div style="background-color: #fff3cd; padding: 20px; border: 1px solid #ffc107; margin-top: 20px; border-radius: 5px;">
                    <h3 style="color: #856404; margin-top: 0; border-bottom: 2px solid #ffc107; padding-bottom: 10px;">Recommended Mitigation Strategies</h3>
                    <ol style="margin: 0; padding-left: 20px;">
        """
        
        for strategy in info['mitigation_strategies']:
            html += f"                        <li style='margin-bottom: 8px;'>{strategy}</li>\n"
        
        html += f"""
                    </ol>
                </div>
                
                <div style="background-color: white; padding: 20px; border: 1px solid #ddd; margin-top: 20px;">
                    <h3 style="color: #1976d2; border-bottom: 2px solid #1976d2; padding-bottom: 10px;">Detection Details</h3>
                    <p><strong>Source File:</strong> {detection_data.get('file', 'N/A')}</p>
                    <p><strong>Total Flows Analyzed:</strong> {detection_data.get('total_flows', 'N/A')}</p>
                    <p><strong>Malicious Flows:</strong> {detection_data.get('malicious_flows', 'N/A')}</p>
                    <p><strong>Detection Timestamp:</strong> {detection_data.get('timestamp', 'N/A')}</p>
                </div>
                
                <div style="background-color: #e3f2fd; padding: 15px; margin-top: 20px; border-radius: 5px; border-left: 4px solid #1976d2;">
                    <p style="margin: 0; font-size: 14px;"><strong>Action Required:</strong> Immediate investigation and implementation of mitigation strategies is recommended to prevent potential system compromise.</p>
                </div>
                
                <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #666; font-size: 12px;">
                    <p>This is an automated alert from the IoT Threat Detection System</p>
                    <p>Generated by Real-Time IoT Security Monitoring Pipeline</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _create_basic_alert(self, detection_data: Dict[str, Any]) -> str:
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2 style="color: #d32f2f;">IoT Threat Detection Alert</h2>
            <p><strong>Attack Type:</strong> {detection_data.get('attack_type', 'Unknown')}</p>
            <p><strong>Confidence:</strong> {detection_data.get('confidence', 'N/A')}</p>
            <p><strong>Timestamp:</strong> {detection_data.get('timestamp', 'N/A')}</p>
            <p><strong>File:</strong> {detection_data.get('file', 'N/A')}</p>
        </body>
        </html>
        """
        return html
    
    def send_alert(self, detection_data: Dict[str, Any]) -> bool:
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"ALERT: {detection_data.get('attack_type', 'Unknown')} Threat Detected"
            msg['From'] = self.email_from
            msg['To'] = self.email_to
            
            html_body = self._create_email_body(detection_data)
            msg.attach(MIMEText(html_body, 'html'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_username, self.email_password)
                server.send_message(msg)
            
            logger.info(f"Alert email sent successfully for {detection_data.get('attack_type')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send alert email: {e}")
            return False
    
    def send_batch_alert(self, detections: List[Dict[str, Any]]) -> bool:
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"ALERT: Multiple Threats Detected ({len(detections)} incidents)"
            msg['From'] = self.email_from
            msg['To'] = self.email_to
            
            html = f"""
            <html>
            <body style="font-family: Arial, sans-serif;">
                <h2 style="color: #d32f2f;">Multiple IoT Threats Detected</h2>
                <p><strong>Total Incidents:</strong> {len(detections)}</p>
                <p><strong>Detection Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <hr>
            """
            
            for idx, detection in enumerate(detections, 1):
                html += f"""
                <div style="margin: 20px 0; padding: 10px; border-left: 3px solid #d32f2f;">
                    <h3>Incident {idx}: {detection.get('attack_type', 'Unknown')}</h3>
                    <p><strong>Confidence:</strong> {detection.get('confidence', 'N/A')}</p>
                    <p><strong>File:</strong> {detection.get('file', 'N/A')}</p>
                </div>
                """
            
            html += """
            </body>
            </html>
            """
            
            msg.attach(MIMEText(html, 'html'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_username, self.email_password)
                server.send_message(msg)
            
            logger.info(f"Batch alert email sent for {len(detections)} detections")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send batch alert email: {e}")
            return False
