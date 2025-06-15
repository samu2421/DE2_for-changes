# test_simple_littles_law.py
"""
Simple test and demo for Little's Law implementation
Run this to see Little's Law in action!
"""

import time
import sys
import os

# Add current directory to path
sys.path.append('.')
sys.path.append('src')

def test_basic_functionality():
    """Test that Little's Law tracking works"""
    print("🧪 Testing Basic Little's Law Functionality")
    print("="*50)
    
    from simple_littles_law import order_arrives, order_completes, show_metrics
    
    # Test 1: Single order
    print("\n1️⃣ Testing single order...")
    order_arrives("TEST_001")
    time.sleep(1)  # Simulate processing time
    order_completes("TEST_001")
    
    # Test 2: Multiple orders
    print("\n2️⃣ Testing multiple orders...")
    orders = ["TEST_002", "TEST_003", "TEST_004"]
    
    # Orders arrive
    for order_id in orders:
        order_arrives(order_id)
        time.sleep(0.2)
    
    # Orders complete
    for order_id in orders:
        time.sleep(0.5)  # Processing time
        order_completes(order_id)
    
    print("\n📊 Final metrics:")
    show_metrics()
    
    print("✅ Basic functionality test completed!")

def demo_realistic_scenario():
    """Demo with realistic e-commerce scenario"""
    print("\n🎬 Realistic E-Commerce Scenario Demo")
    print("="*50)
    
    from simple_littles_law import tracker
    
    # Clear any existing data
    tracker.orders_in_system.clear()
    tracker.completed_orders.clear()
    
    print("Simulating realistic order flow...")
    
    # Simulate orders arriving at different rates
    scenarios = [
        ("Morning Rush", 0.1, 5),     # 5 orders, fast arrival
        ("Lunch Break", 0.3, 3),     # 3 orders, slower arrival  
        ("Afternoon Peak", 0.1, 8),  # 8 orders, fast arrival
        ("Evening", 0.5, 2)          # 2 orders, slow arrival
    ]
    
    all_orders = []
    
    for scenario_name, arrival_delay, num_orders in scenarios:
        print(f"\n📈 {scenario_name}: {num_orders} orders")
        
        # Orders arrive
        for i in range(num_orders):
            order_id = f"{scenario_name.upper().replace(' ', '_')}_{i:03d}"
            tracker.order_arrives(order_id)
            all_orders.append(order_id)
            time.sleep(arrival_delay)
        
        # Show current state
        metrics = tracker.calculate_metrics()
        print(f"   Current WIP: {metrics['L_work_in_progress']} orders")
    
    print(f"\n⏳ Processing all {len(all_orders)} orders...")
    
    # Process orders with varying completion times
    for i, order_id in enumerate(all_orders):
        # Simulate different processing times
        processing_time = 0.3 + (i % 3) * 0.2  # Vary between 0.3-0.7 seconds
        time.sleep(processing_time)
        tracker.order_completes(order_id)
        
        # Show metrics every 5 completions
        if (i + 1) % 5 == 0:
            print(f"\nAfter {i+1} completions:")
            tracker.print_metrics()
    
    print(f"\n🎉 Demo completed! Processed {len(all_orders)} orders")

def test_integration_points():
    """Test integration with existing project components"""
    print("\n🔌 Testing Integration Points")
    print("="*30)
    
    # Test 1: Can import from existing project structure
    try:
        from simple_littles_law import LittlesLaw
        print("✅ Can import LittlesLaw class")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 2: Works with JSON data (like your streaming)
    try:
        import json
        
        # Simulate order data like your streaming consumer receives
        order_data = {
            'InvoiceNo': 'INTEGRATION_TEST_001',
            'Quantity': 3,
            'UnitPrice': 15.99,
            'Country': 'United Kingdom'
        }
        
        tracker = LittlesLaw()
        tracker.order_arrives(order_data['InvoiceNo'])
        time.sleep(0.5)
        tracker.order_completes(order_data['InvoiceNo'])
        
        print("✅ Works with JSON order data")
    except Exception as e:
        print(f"❌ JSON integration failed: {e}")
        return False
    
    # Test 3: Generates reports
    try:
        report = tracker.get_simple_report()
        assert 'current_metrics' in report
        assert 'system_status' in report
        print("✅ Report generation works")
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        return False
    
    print("✅ All integration tests passed!")
    return True

def quick_demo():
    """Super quick demo for immediate results"""
    print("⚡ QUICK LITTLE'S LAW DEMO")
    print("="*30)
    
    from simple_littles_law import order_arrives, order_completes, show_metrics
    
    # Rapid fire demo
    orders = [f"QUICK_{i:03d}" for i in range(3)]
    
    print("📦 Orders arriving...")
    for order_id in orders:
        order_arrives(order_id)
        time.sleep(0.1)
    
    print("⚡ Processing orders...")
    for order_id in orders:
        time.sleep(0.3)
        order_completes(order_id)
    
    print("📊 Results:")
    show_metrics()

def main():
    """Main test function"""
    print("🎯 SIMPLE LITTLE'S LAW - TESTING & DEMO")
    print("="*60)
    
    # Check if user wants quick demo
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        quick_demo()
        return
    
    print("Choose what to run:")
    print("1. Quick demo (30 seconds)")
    print("2. Basic functionality test")
    print("3. Realistic scenario demo")
    print("4. Integration tests")
    print("5. All tests")
    
    try:
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            quick_demo()
            
        elif choice == "2":
            test_basic_functionality()
            
        elif choice == "3":
            demo_realistic_scenario()
            
        elif choice == "4":
            test_integration_points()
            
        elif choice == "5":
            print("🚀 Running all tests...\n")
            test_basic_functionality()
            demo_realistic_scenario()
            test_integration_points()
            print("\n🎉 All tests completed!")
            
        else:
            print("Invalid choice, running quick demo...")
            quick_demo()
    
    except KeyboardInterrupt:
        print("\n🛑 Demo cancelled")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()