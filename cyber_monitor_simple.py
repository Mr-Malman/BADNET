#!/usr/bin/env python3
"""
Cyber Attack Real-Time Monitor 
=============================================
"""

import argparse
import time
import numpy as np
import pandas as pd
import joblib
import psutil
import subprocess
from datetime import datetime
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class SimpleCyberMonitor:
    def __init__(self, model_path=None, scaler_path=None):
        """Initialize the simplified cyber attack monitor."""
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.attack_count = 0
        self.total_predictions = 0
        self.alert_history = deque(maxlen=50)
        
        # Load the trained model and preprocessors
        self.load_model(model_path, scaler_path)
        
        # Statistics tracking
        self.stats = {
            'cpu_history': deque(maxlen=20),
            'memory_history': deque(maxlen=20),
            'network_history': deque(maxlen=20)
        }
    
    def load_model(self, model_path=None, scaler_path=None):
        """Load the trained model and preprocessors."""
        try:
            # Default paths
            if model_path is None:
                model_path = "processed_data/supervised_learning/best_model.pkl"
            if scaler_path is None:
                scaler_path = "processed_data/supervised_learning/feature_scaler.pkl"
            
            # Load feature names
            feature_file = "processed_data/supervised_learning/train_features.txt"
            with open(feature_file, 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]
            
            print(f"Loaded {len(self.feature_names)} features")
            
            # Load scaler
            try:
                self.scaler = joblib.load(scaler_path)
                print("Loaded feature scaler")
            except FileNotFoundError:
                print(" Scaler not found, using default values")
            
            # Load the trained model
            try:
                self.model = joblib.load(model_path)
                print(f"Loaded trained model: {type(self.model).__name__}")
            except FileNotFoundError:
                print(" Trained model not found, using Logistic Regression")
                from sklearn.linear_model import LogisticRegression
                self.model = LogisticRegression(random_state=42)
                
        except Exception as e:
            print(f"Error loading model: {e}")
            print(" Running in simulation mode")
    
    def get_system_features(self):
        """Get basic system features."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            
            # Process count
            process_count = len(psutil.pids())
            
            # Network I/O (basic)
            net_io = psutil.net_io_counters()
            
            features = {
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'memory_available': memory.available / (1024**3),  # GB
                'disk_read_bytes': disk_io.read_bytes if disk_io else 0,
                'disk_write_bytes': disk_io.write_bytes if disk_io else 0,
                'process_count': process_count,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'high_cpu': 1 if cpu_percent > 80 else 0,
                'high_memory': 1 if memory.percent > 80 else 0,
                'high_traffic': 1 if (net_io.packets_sent + net_io.packets_recv) > 10000 else 0,
                'disk_activity': 1 if disk_io and (disk_io.read_bytes + disk_io.write_bytes) > 1000000 else 0,
            }
            
            return features
            
        except Exception as e:
            print(f"Error getting system features: {e}")
            return {}
    
    def get_network_features(self):
        """Get basic network features (without requiring special permissions)."""
        try:
            # Basic network statistics
            net_io = psutil.net_io_counters()
            
            # Try to get connection count (may fail on macOS)
            try:
                connections = psutil.net_connections()
                connection_count = len([conn for conn in connections if conn.status == 'ESTABLISHED'])
            except:
                connection_count = 0
            
            features = {
                'connection_count': connection_count,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packet_rate': (net_io.packets_sent + net_io.packets_recv) / max(1, connection_count),
                'byte_rate': (net_io.bytes_sent + net_io.bytes_recv) / max(1, connection_count),
                'high_traffic': 1 if (net_io.packets_sent + net_io.packets_recv) > 10000 else 0,
                'suspicious_ports': 0,  # Simplified for macOS
            }
            
            return features
            
        except Exception as e:
            print(f"Error getting network features: {e}")
            return {}
    
    def preprocess_features(self, features_dict):
        """Preprocess features to match the training data format."""
        try:
            # Create a DataFrame with the expected features
            df = pd.DataFrame([features_dict])
            
            # Add missing features with default values
            for feature in self.feature_names:
                if feature not in df.columns:
                    df[feature] = 0
            
            # Ensure correct order
            df = df[self.feature_names]
            
            # Apply preprocessing
            if self.scaler:
                df_scaled = self.scaler.transform(df)
            else:
                df_scaled = df.values
            
            return df_scaled
            
        except Exception as e:
            print(f"Error preprocessing features: {e}")
            return None
    
    def predict_attack(self, features):
        """Predict if current activity indicates an attack."""
        try:
            if self.model is None:
                return False, 0.0
            
            # Make prediction
            prediction = self.model.predict(features)
            probability = self.model.predict_proba(features)[0][1] if hasattr(self.model, 'predict_proba') else 0.5
            
            return bool(prediction[0]), probability
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return False, 0.0
    
    def generate_alert(self, attack_detected, probability, features, mode):
        """Generate and log security alerts."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if attack_detected:
            self.attack_count += 1
            alert_level = "CRITICAL" if probability > 0.8 else " WARNING"
            
            alert = {
                'timestamp': timestamp,
                'level': alert_level,
                'probability': f"{probability:.2%}",
                'mode': mode
            }
            
            self.alert_history.append(alert)
            
            print(f"\n{alert_level} CYBER ATTACK DETECTED!")
            print(f"Time: {timestamp}")
            print(f"Probability: {probability:.2%}")
            print(f"Mode: {mode}")
            print(f"Suspicious Indicators:")
            
            # Highlight suspicious features
            for key, value in features.items():
                if isinstance(value, (int, float)) and value > 0:
                    if key in ['high_cpu', 'high_memory', 'high_traffic', 'disk_activity']:
                        print(f"   • {key}: {value}")
            
            print("-" * 50)
        
        self.total_predictions += 1
    
    def monitor_system(self, duration=None, interval=5):
        """Monitor system activity."""
        print(f"💻 Starting system monitoring (interval: {interval}s)")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            while True:
                if duration and (time.time() - start_time) > duration:
                    break
                
                # Get system features
                features = self.get_system_features()
                
                if features:
                    # Preprocess features
                    processed_features = self.preprocess_features(features)
                    
                    if processed_features is not None:
                        # Make prediction
                        attack_detected, probability = self.predict_attack(processed_features)
                        
                        # Generate alert if needed
                        self.generate_alert(attack_detected, probability, features, "SYSTEM")
                
                # Display status
                self.display_status("SYSTEM", features)
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nSystem monitoring stopped")
    
    def monitor_network(self, duration=None, interval=5):
        """Monitor network activity (basic)."""
        print(f"Starting network monitoring (interval: {interval}s)")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            while True:
                if duration and (time.time() - start_time) > duration:
                    break
                
                # Get network features
                features = self.get_network_features()
                
                if features:
                    # Preprocess features
                    processed_features = self.preprocess_features(features)
                    
                    if processed_features is not None:
                        # Make prediction
                        attack_detected, probability = self.predict_attack(processed_features)
                        
                        # Generate alert if needed
                        self.generate_alert(attack_detected, probability, features, "NETWORK")
                
                # Display status
                self.display_status("NETWORK", features)
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n⏹Network monitoring stopped")
    
    def monitor_both(self, duration=None, interval=5):
        """Monitor both system and network activity."""
        print(f"Starting comprehensive monitoring (interval: {interval}s)")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            while True:
                if duration and (time.time() - start_time) > duration:
                    break
                
                # Get both types of features
                system_features = self.get_system_features()
                network_features = self.get_network_features()
                
                # Combine features
                combined_features = {**system_features, **network_features}
                
                if combined_features:
                    # Preprocess features
                    processed_features = self.preprocess_features(combined_features)
                    
                    if processed_features is not None:
                        # Make prediction
                        attack_detected, probability = self.predict_attack(processed_features)
                        
                        # Generate alert if needed
                        self.generate_alert(attack_detected, probability, combined_features, "COMPREHENSIVE")
                
                # Display status
                self.display_status("COMPREHENSIVE", combined_features)
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nComprehensive monitoring stopped")
    
    def display_status(self, mode, features):
        """Display current monitoring status."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Extract key metrics
        cpu_usage = features.get('cpu_usage', 0)
        memory_usage = features.get('memory_usage', 0)
        connection_count = features.get('connection_count', 0)
        packet_rate = features.get('packet_rate', 0)
        
        # Display status line
        status_line = f"[{timestamp}] "
        
        if mode == "SYSTEM":
            status_line += f"CPU: {cpu_usage:.1f}% | Memory: {memory_usage:.1f}%"
        elif mode == "NETWORK":
            status_line += f"Connections: {connection_count} | Packet Rate: {packet_rate:.1f}/s"
        elif mode == "COMPREHENSIVE":
            status_line += f"CPU: {cpu_usage:.1f}% | Mem: {memory_usage:.1f}% | Conn: {connection_count}"
        
        status_line += f" | Attacks: {self.attack_count}/{self.total_predictions}"
        
        print(f"\r{status_line}", end="", flush=True)
    
    def show_summary(self):
        """Show monitoring summary."""
        print("\n" + "=" * 60)
        print("MONITORING SUMMARY")
        print("=" * 60)
        print(f"Total Predictions: {self.total_predictions}")
        print(f"Attacks Detected: {self.attack_count}")
        print(f"Detection Rate: {self.attack_count/max(1, self.total_predictions)*100:.2f}%")
        
        if self.alert_history:
            print(f"\nRecent Alerts ({len(self.alert_history)}):")
            for alert in list(self.alert_history)[-5:]:
                print(f"  {alert['timestamp']} - {alert['level']} ({alert['probability']})")
        
        print("=" * 60)

def main():
    """Main function for the simplified cyber attack monitor."""
    parser = argparse.ArgumentParser(
        description="Cyber Attack Real-Time Monitor (Simplified)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 cyber_monitor_simple.py --mode system --interval 10
  python3 cyber_monitor_simple.py --mode network --duration 300
  python3 cyber_monitor_simple.py --mode both --interval 5 --duration 600
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['network', 'system', 'both'],
        default='both',
        help='Monitoring mode (default: both)'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=5,
        help='Monitoring interval in seconds (default: 5)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        help='Monitoring duration in seconds (default: continuous)'
    )
    
    parser.add_argument(
        '--model',
        help='Path to trained model file'
    )
    
    parser.add_argument(
        '--scaler',
        help='Path to feature scaler file'
    )
    
    args = parser.parse_args()
    
    # Initialize monitor
    print("Cyber Attack Real-Time Monitor (Simplified)")
    print("=" * 50)
    
    monitor = SimpleCyberMonitor(
        model_path=args.model,
        scaler_path=args.scaler
    )
    
    try:
        # Start monitoring based on mode
        if args.mode == 'network':
            monitor.monitor_network(args.duration, args.interval)
        elif args.mode == 'system':
            monitor.monitor_system(args.duration, args.interval)
        elif args.mode == 'both':
            monitor.monitor_both(args.duration, args.interval)
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
    
    finally:
        # Show summary
        monitor.show_summary()

if __name__ == "__main__":
    main()
