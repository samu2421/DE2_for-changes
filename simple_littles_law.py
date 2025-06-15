# src/simple_littles_law.py
"""
Simple Little's Law Implementation for E-Commerce Analytics
Just tracks orders and calculates L = λ × W

Usage:
    from simple_littles_law import LittlesLaw
    
    tracker = LittlesLaw()
    tracker.order_arrives("ORDER_001")
    tracker.order_completes("ORDER_001")
    tracker.print_metrics()
"""

import time
from datetime import datetime, timedelta
from collections import defaultdict
import json

class LittlesLaw:
    """
    Simple Little's Law tracker
    L = λ × W (Work in Progress = Arrival Rate × Lead Time)
    """
    
    def __init__(self):
        # Track orders currently in system
        self.orders_in_system = {}  # order_id -> arrival_time
        
        # Track completed orders
        self.completed_orders = []  # list of {order_id, arrival_time, completion_time, lead_time}
        
        # Keep only recent data (last 1 hour)
        self.time_window_minutes = 60
    
    def order_arrives(self, order_id, metadata=None):
        """Record when an order arrives in the system"""
        arrival_time = datetime.now()
        self.orders_in_system[order_id] = arrival_time
        print(f"📦 Order {order_id} arrived")
    
    def order_completes(self, order_id):
        """Record when an order completes processing"""
        completion_time = datetime.now()
        
        if order_id in self.orders_in_system:
            arrival_time = self.orders_in_system[order_id]
            lead_time_minutes = (completion_time - arrival_time).total_seconds() / 60.0
            
            # Record completion
            self.completed_orders.append({
                'order_id': order_id,
                'arrival_time': arrival_time,
                'completion_time': completion_time,
                'lead_time_minutes': lead_time_minutes
            })
            
            # Remove from active orders
            del self.orders_in_system[order_id]
            
            print(f"✅ Order {order_id} completed in {lead_time_minutes:.2f} minutes")
        else:
            print(f"⚠️ Order {order_id} not found in system")
    
    def calculate_metrics(self):
        """Calculate Little's Law metrics: L, λ, W"""
        now = datetime.now()
        window_start = now - timedelta(minutes=self.time_window_minutes)
        
        # L = Work in Progress (orders currently in system)
        L = len(self.orders_in_system)
        
        # λ = Arrival Rate (orders per minute in the time window)
        recent_arrivals = [
            order for order in self.completed_orders 
            if order['arrival_time'] > window_start
        ]
        lambda_rate = len(recent_arrivals) / self.time_window_minutes if recent_arrivals else 0
        
        # W = Lead Time (average processing time in minutes)
        recent_completions = [
            order for order in self.completed_orders 
            if order['completion_time'] > window_start
        ]
        
        if recent_completions:
            W = sum(order['lead_time_minutes'] for order in recent_completions) / len(recent_completions)
        else:
            W = 0
        
        # Little's Law: L should equal λ × W
        predicted_L = lambda_rate * W
        difference = abs(L - predicted_L)
        
        return {
            'L_work_in_progress': L,
            'lambda_arrival_rate': round(lambda_rate, 3),
            'W_lead_time_minutes': round(W, 2),
            'littles_law_predicted': round(predicted_L, 2),
            'difference': round(difference, 2),
            'is_balanced': difference < 1.0,
            'timestamp': now.isoformat()
        }
    
    def print_metrics(self):
        """Print current Little's Law metrics"""
        metrics = self.calculate_metrics()
        
        print("\n" + "="*50)
        print("📊 LITTLE'S LAW METRICS")
        print("="*50)
        print(f"L (Work in Progress): {metrics['L_work_in_progress']} orders")
        print(f"λ (Arrival Rate): {metrics['lambda_arrival_rate']} orders/minute")
        print(f"W (Lead Time): {metrics['W_lead_time_minutes']} minutes")
        print(f"λ × W (Predicted): {metrics['littles_law_predicted']} orders")
        print(f"Difference: {metrics['difference']}")
        
        if metrics['is_balanced']:
            print("✅ System is BALANCED (Little's Law holds)")
        else:
            print("⚠️ System may be OUT OF BALANCE")
        
        print("="*50)
        
        return metrics
    
    def get_simple_report(self):
        """Get a simple status report"""
        metrics = self.calculate_metrics()
        total_completed = len(self.completed_orders)
        
        return {
            'current_metrics': metrics,
            'total_orders_processed': total_completed,
            'orders_currently_processing': len(self.orders_in_system),
            'system_status': 'balanced' if metrics['is_balanced'] else 'imbalanced'
        }

# Global tracker instance for easy use
tracker = LittlesLaw()

# Convenience functions
def order_arrives(order_id, metadata=None):
    """Simple function to record order arrival"""
    tracker.order_arrives(order_id, metadata)

def order_completes(order_id):
    """Simple function to record order completion"""
    tracker.order_completes(order_id)

def show_metrics():
    """Simple function to show current metrics"""
    return tracker.print_metrics()

def get_report():
    """Simple function to get report"""
    return tracker.get_simple_report()

# Demo function
def demo():
    """Run a simple demonstration"""
    print("🎯 Simple Little's Law Demo")
    print("Simulating 5 orders...")
    
    # Simulate orders arriving
    for i in range(5):
        order_id = f"DEMO_{i:03d}"
        order_arrives(order_id)
        time.sleep(0.2)  # Small delay between arrivals
    
    print(f"\n⏳ Processing orders...")
    
    # Simulate orders completing
    for i in range(5):
        order_id = f"DEMO_{i:03d}"
        time.sleep(0.5)  # Simulate processing time
        order_completes(order_id)
    
    print(f"\n📊 Final Results:")
    show_metrics()

if __name__ == "__main__":
    demo()