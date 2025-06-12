# src/monitoring/monitor.py
import time
import psutil
import requests
import json
from datetime import datetime
import logging
from pathlib import Path
import threading
import sys
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('docs/monitoring_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EcommerceSystemMonitor:
    """
    Comprehensive system monitoring for the e-commerce analytics platform
    Monitors API health, system performance, and business metrics
    """
    
    def __init__(self, ml_api_url='http://localhost:5001'):
        self.ml_api_url = ml_api_url
        self.monitoring_active = False
        self.metrics_history = []
        self.alerts = []
        
        # Performance thresholds
        self.thresholds = {
            'cpu_warning': 80.0,
            'cpu_critical': 95.0,
            'memory_warning': 80.0,
            'memory_critical': 95.0,
            'api_response_warning': 1000,  # ms
            'api_response_critical': 5000  # ms
        }
        
        # Create monitoring directories
        Path('docs').mkdir(exist_ok=True)
        Path('data/monitoring').mkdir(exist_ok=True)
    
    def check_system_health(self):
        """Check overall system health"""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'system': self.get_system_metrics(),
            'api': self.check_api_health(),
            'data': self.check_data_health(),
            'alerts': []
        }
        
        # Generate alerts based on thresholds
        self.generate_alerts(health_status)
        
        return health_status
    
    def get_system_metrics(self):
        """Get system performance metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024**3)
            
            # Network metrics (if available)
            try:
                network = psutil.net_io_counters()
                network_stats = {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                }
            except:
                network_stats = {'error': 'Network stats unavailable'}
            
            # Process count
            process_count = len(psutil.pids())
            
            system_metrics = {
                'cpu': {
                    'percent': round(cpu_percent, 2),
                    'count': cpu_count,
                    'status': self.get_status_level(cpu_percent, 'cpu')
                },
                'memory': {
                    'percent': round(memory_percent, 2),
                    'available_gb': round(memory_available_gb, 2),
                    'total_gb': round(memory.total / (1024**3), 2),
                    'status': self.get_status_level(memory_percent, 'memory')
                },
                'disk': {
                    'percent': round(disk_percent, 2),
                    'free_gb': round(disk_free_gb, 2),
                    'total_gb': round(disk.total / (1024**3), 2)
                },
                'network': network_stats,
                'processes': process_count
            }
            
            return system_metrics
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {'error': str(e)}
    
    def check_api_health(self):
        """Check ML API health and performance"""
        api_health = {
            'status': 'unknown',
            'response_time_ms': None,
            'last_check': datetime.now().isoformat(),
            'prediction_test': None
        }
        
        try:
            # Health check
            start_time = time.time()
            response = requests.get(f"{self.ml_api_url}/health", timeout=10)
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            api_health['response_time_ms'] = round(response_time, 2)
            
            if response.status_code == 200:
                api_health['status'] = 'healthy'
                
                # Test prediction endpoint
                test_prediction = {
                    'quantity': 2,
                    'unit_price': 15.99,
                    'hour': 12,
                    'day_of_week': 1
                }
                
                pred_start = time.time()
                pred_response = requests.post(
                    f"{self.ml_api_url}/predict", 
                    json=test_prediction, 
                    timeout=5
                )
                pred_time = (time.time() - pred_start) * 1000
                
                if pred_response.status_code == 200:
                    pred_result = pred_response.json()
                    api_health['prediction_test'] = {
                        'status': 'success',
                        'response_time_ms': round(pred_time, 2),
                        'predicted_revenue': pred_result.get('predicted_revenue', 0)
                    }
                else:
                    api_health['prediction_test'] = {
                        'status': 'error',
                        'error_code': pred_response.status_code
                    }
            else:
                api_health['status'] = 'error'
                api_health['error_code'] = response.status_code
                
        except requests.exceptions.RequestException as e:
            api_health['status'] = 'offline'
            api_health['error'] = str(e)
        except Exception as e:
            api_health['status'] = 'error'
            api_health['error'] = str(e)
        
        return api_health
    
    def check_data_health(self):
        """Check data files and pipeline health"""
        data_health = {
            'raw_data': self.check_file_health('data/OnlineRetail.csv'),
            'processed_data': self.check_directory_health('data/processed'),
            'ml_models': self.check_ml_files_health(),
            'logs': self.check_file_health('docs/batch_processing.log')
        }
        
        return data_health
    
    def check_file_health(self, file_path):
        """Check individual file health"""
        path = Path(file_path)
        if path.exists():
            stat = path.stat()
            return {
                'exists': True,
                'size_mb': round(stat.st_size / (1024**2), 2),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'status': 'healthy'
            }
        else:
            return {
                'exists': False,
                'status': 'missing'
            }
    
    def check_directory_health(self, dir_path):
        """Check directory health"""
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            files = list(path.glob('*'))
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            
            return {
                'exists': True,
                'file_count': len(files),
                'total_size_mb': round(total_size / (1024**2), 2),
                'status': 'healthy'
            }
        else:
            return {
                'exists': False,
                'status': 'missing'
            }
    
    def check_ml_files_health(self):
        """Check ML model files health"""
        ml_files = [
            'src/ml/revenue_model.pkl',
            'src/ml/feature_scaler.pkl',
            'src/ml/label_encoders.pkl',
            'src/ml/feature_names.txt'
        ]
        
        ml_health = {}
        for file_path in ml_files:
            file_name = Path(file_path).name
            ml_health[file_name] = self.check_file_health(file_path)
        
        return ml_health
    
    def get_status_level(self, value, metric_type):
        """Get status level based on thresholds"""
        warning_threshold = self.thresholds.get(f'{metric_type}_warning', 80)
        critical_threshold = self.thresholds.get(f'{metric_type}_critical', 95)
        
        if value >= critical_threshold:
            return 'critical'
        elif value >= warning_threshold:
            return 'warning'
        else:
            return 'healthy'
    
    def generate_alerts(self, health_status):
        """Generate alerts based on health status"""
        alerts = []
        
        # System alerts
        if 'system' in health_status:
            system = health_status['system']
            
            if 'cpu' in system and system['cpu']['status'] in ['warning', 'critical']:
                alerts.append({
                    'type': 'system',
                    'level': system['cpu']['status'],
                    'message': f"High CPU usage: {system['cpu']['percent']}%",
                    'timestamp': health_status['timestamp']
                })
            
            if 'memory' in system and system['memory']['status'] in ['warning', 'critical']:
                alerts.append({
                    'type': 'system',
                    'level': system['memory']['status'],
                    'message': f"High memory usage: {system['memory']['percent']}%",
                    'timestamp': health_status['timestamp']
                })
        
        # API alerts
        if 'api' in health_status:
            api = health_status['api']
            
            if api['status'] == 'offline':
                alerts.append({
                    'type': 'api',
                    'level': 'critical',
                    'message': "ML API is offline",
                    'timestamp': health_status['timestamp']
                })
            elif api['status'] == 'error':
                alerts.append({
                    'type': 'api',
                    'level': 'warning',
                    'message': f"ML API error: {api.get('error', 'Unknown error')}",
                    'timestamp': health_status['timestamp']
                })
        
        health_status['alerts'] = alerts
        self.alerts.extend(alerts)
    
    def save_metrics(self, health_status):
        """Save metrics to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save current metrics
        metrics_file = f'data/monitoring/metrics_{timestamp}.json'
        with open(metrics_file, 'w') as f:
            json.dump(health_status, f, indent=2)
        
        # Keep metrics history
        self.metrics_history.append(health_status)
        
        # Keep only last 100 metrics to prevent memory issues
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
    
    def print_dashboard(self, health_status):
        """Print monitoring dashboard"""
        print("\n" + "="*60)
        print("🖥️  E-COMMERCE SYSTEM MONITORING DASHBOARD")
        print("="*60)
        print(f"🕐 Timestamp: {health_status['timestamp']}")
        
        # System metrics
        if 'system' in health_status and 'error' not in health_status['system']:
            sys_metrics = health_status['system']
            print(f"\n💻 SYSTEM PERFORMANCE:")
            print(f"   CPU: {sys_metrics['cpu']['percent']}% ({sys_metrics['cpu']['status']})")
            print(f"   Memory: {sys_metrics['memory']['percent']}% ({sys_metrics['memory']['status']})")
            print(f"   Disk: {sys_metrics['disk']['percent']}% used")
            print(f"   Processes: {sys_metrics['processes']}")
        
        # API health
        if 'api' in health_status:
            api = health_status['api']
            status_emoji = {
                'healthy': '✅',
                'warning': '⚠️',
                'error': '❌',
                'offline': '🔴',
                'unknown': '❓'
            }
            
            print(f"\n🤖 ML API STATUS:")
            print(f"   Status: {status_emoji.get(api['status'], '❓')} {api['status'].upper()}")
            
            if api['response_time_ms']:
                print(f"   Response Time: {api['response_time_ms']}ms")
            
            if api.get('prediction_test'):
                pred_test = api['prediction_test']
                if pred_test['status'] == 'success':
                    print(f"   Prediction Test: ✅ £{pred_test['predicted_revenue']:.2f} ({pred_test['response_time_ms']}ms)")
                else:
                    print(f"   Prediction Test: ❌ Failed")
        
        # Data health
        if 'data' in health_status:
            data = health_status['data']
            print(f"\n📊 DATA HEALTH:")
            
            if data['raw_data']['exists']:
                print(f"   Raw Data: ✅ {data['raw_data']['size_mb']} MB")
            else:
                print(f"   Raw Data: ❌ Missing")
            
            if data['processed_data']['exists']:
                print(f"   Processed: ✅ {data['processed_data']['file_count']} files")
            else:
                print(f"   Processed: ❌ Missing")
            
            # ML models
            ml_models = data['ml_models']
            model_count = sum(1 for model in ml_models.values() if model['exists'])
            total_models = len(ml_models)
            print(f"   ML Models: {'✅' if model_count == total_models else '⚠️'} {model_count}/{total_models} files")
        
        # Alerts
        if health_status['alerts']:
            print(f"\n🚨 ACTIVE ALERTS:")
            for alert in health_status['alerts']:
                level_emoji = {'warning': '⚠️', 'critical': '🔥'}
                print(f"   {level_emoji.get(alert['level'], '⚠️')} {alert['message']}")
        else:
            print(f"\n✅ NO ACTIVE ALERTS")
        
        print("="*60)
    
    def start_monitoring(self, interval=30, duration=None):
        """Start continuous monitoring"""
        print(f"🚀 Starting system monitoring (interval: {interval}s)")
        print("Press Ctrl+C to stop monitoring")
        
        self.monitoring_active = True
        start_time = time.time()
        
        try:
            while self.monitoring_active:
                # Check health
                health_status = self.check_system_health()
                
                # Save metrics
                self.save_metrics(health_status)
                
                # Print dashboard
                self.print_dashboard(health_status)
                
                # Check duration limit
                if duration and (time.time() - start_time) >= duration:
                    print(f"\n⏰ Monitoring duration ({duration}s) reached")
                    break
                
                # Wait for next interval
                print(f"\n💤 Waiting {interval} seconds for next check...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print(f"\n🛑 Monitoring stopped by user")
        finally:
            self.monitoring_active = False
            self.generate_monitoring_report()
    
    def generate_monitoring_report(self):
        """Generate final monitoring report"""
        if not self.metrics_history:
            return
        
        print(f"\n📋 MONITORING SESSION REPORT")
        print("="*40)
        
        # Calculate averages
        cpu_values = [m['system']['cpu']['percent'] for m in self.metrics_history if 'system' in m and 'cpu' in m['system']]
        memory_values = [m['system']['memory']['percent'] for m in self.metrics_history if 'system' in m and 'memory' in m['system']]
        
        if cpu_values:
            print(f"Average CPU: {sum(cpu_values)/len(cpu_values):.1f}%")
            print(f"Max CPU: {max(cpu_values):.1f}%")
        
        if memory_values:
            print(f"Average Memory: {sum(memory_values)/len(memory_values):.1f}%")
            print(f"Max Memory: {max(memory_values):.1f}%")
        
        # Alert summary
        total_alerts = len(self.alerts)
        if total_alerts > 0:
            print(f"Total Alerts: {total_alerts}")
            
            # Group alerts by type
            alert_types = {}
            for alert in self.alerts:
                alert_type = alert['type']
                alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
            
            for alert_type, count in alert_types.items():
                print(f"  {alert_type}: {count}")
        else:
            print("Total Alerts: 0 ✅")
        
        print(f"Session Duration: {len(self.metrics_history)} checks")
        
        # Save final report
        report = {
            'session_summary': {
                'total_checks': len(self.metrics_history),
                'avg_cpu': sum(cpu_values)/len(cpu_values) if cpu_values else 0,
                'max_cpu': max(cpu_values) if cpu_values else 0,
                'avg_memory': sum(memory_values)/len(memory_values) if memory_values else 0,
                'max_memory': max(memory_values) if memory_values else 0,
                'total_alerts': total_alerts,
                'alert_breakdown': alert_types if total_alerts > 0 else {}
            },
            'all_metrics': self.metrics_history
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f'data/monitoring/session_report_{timestamp}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📁 Report saved to: {report_file}")

def main():
    """Main monitoring function"""
    print("🖥️ E-COMMERCE SYSTEM MONITOR")
    print("="*40)
    
    monitor = EcommerceSystemMonitor()
    
    # Show options
    print("Choose monitoring mode:")
    print("1. Single health check")
    print("2. Continuous monitoring (30s intervals)")
    print("3. Quick monitoring (5 checks)")
    print("4. Custom monitoring")
    
    try:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            print("\n🔍 Running single health check...")
            health = monitor.check_system_health()
            monitor.print_dashboard(health)
            
        elif choice == "2":
            monitor.start_monitoring(interval=30)
            
        elif choice == "3":
            print("\n⚡ Running quick monitoring (5 checks)...")
            for i in range(5):
                print(f"\n--- Check {i+1}/5 ---")
                health = monitor.check_system_health()
                monitor.print_dashboard(health)
                if i < 4:  # Don't wait after last check
                    time.sleep(10)
            
        elif choice == "4":
            interval = int(input("Enter monitoring interval (seconds): "))
            duration = input("Enter duration (seconds, or press Enter for unlimited): ").strip()
            duration = int(duration) if duration else None
            
            monitor.start_monitoring(interval=interval, duration=duration)
            
        else:
            print("Invalid choice, running single health check...")
            health = monitor.check_system_health()
            monitor.print_dashboard(health)
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring cancelled")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    print("\n✅ Monitoring session complete!")

if __name__ == "__main__":
    main()
