# setup_project.py
import os
from pathlib import Path
import subprocess
import sys

def create_directories():
    """Create all necessary project directories"""
    directories = [
        'data/cleaned',
        'data/monitoring', 
        'data/test_data',
        'docs',
        'src/monitoring',
        'logs'
    ]
    
    print("📁 Creating project directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created: {directory}")

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'faker', 'matplotlib', 'seaborn',
        'scikit-learn', 'flask', 'requests', 'psutil'
    ]
    
    optional_packages = [
        'google-cloud-pubsub', 'google-cloud-bigquery'
    ]
    
    print("\n📦 Checking required dependencies...")
    missing_required = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            missing_required.append(package)
            print(f"   ❌ {package}")
    
    print("\n📦 Checking optional dependencies (for streaming)...")
    missing_optional = []
    
    for package in optional_packages:
        try:
            if package == 'google-cloud-pubsub':
                from google.cloud import pubsub_v1
                print(f"   ✅ {package}")
            elif package == 'google-cloud-bigquery':
                from google.cloud import bigquery
                print(f"   ✅ {package}")
        except ImportError:
            missing_optional.append(package)
            print(f"   ⚠️ {package} (optional)")
    
    if missing_required:
        print(f"\n🚨 Missing required packages: {', '.join(missing_required)}")
        print("Install with: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        print(f"\n💡 Missing optional packages: {', '.join(missing_optional)}")
        print("For streaming features, install with: pip install " + " ".join(missing_optional))
    
    return True

def check_data_files():
    """Check if essential data files exist"""
    print("\n📊 Checking data files...")
    
    essential_files = {
        'data/OnlineRetail.csv': 'Main dataset from Kaggle',
    }
    
    optional_files = {
        'src/ml/revenue_model.pkl': 'Trained ML model',
        'src/ml/feature_scaler.pkl': 'Feature scaler',
        'src/ml/label_encoders.pkl': 'Label encoders'
    }
    
    missing_essential = []
    
    for file_path, description in essential_files.items():
        if Path(file_path).exists():
            size_mb = Path(file_path).stat().st_size / (1024*1024)
            print(f"   ✅ {file_path} ({size_mb:.1f} MB)")
        else:
            missing_essential.append((file_path, description))
            print(f"   ❌ {file_path} - {description}")
    
    for file_path, description in optional_files.items():
        if Path(file_path).exists():
            size_kb = Path(file_path).stat().st_size / 1024
            print(f"   ✅ {file_path} ({size_kb:.1f} KB)")
        else:
            print(f"   ⚠️ {file_path} - {description} (will be created)")
    
    if missing_essential:
        print(f"\n🚨 Missing essential files:")
        for file_path, description in missing_essential:
            print(f"   • {file_path}: {description}")
        
        if 'data/OnlineRetail.csv' in [f[0] for f in missing_essential]:
            print("\n💡 Download OnlineRetail.csv from:")
            print("   https://www.kaggle.com/datasets/vijayuv/onlineretail")
        
        return False
    
    return True

def run_initial_setup():
    """Run initial setup tasks"""
    print("\n🚀 Running initial setup tasks...")
    
    # 1. Data analysis (if data exists)
    if Path('data/OnlineRetail.csv').exists():
        print("\n1️⃣ Running data analysis...")
        try:
            result = subprocess.run(['python', 'src/data_analysis.py'], 
                                  capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                print("   ✅ Data analysis completed")
            else:
                print(f"   ⚠️ Data analysis had issues: {result.stderr[:100]}...")
        except Exception as e:
            print(f"   ❌ Could not run data analysis: {e}")
    
    # 2. Train ML model (if data exists)
    if Path('data/OnlineRetail.csv').exists() and not Path('src/ml/revenue_model.pkl').exists():
        print("\n2️⃣ Training ML model...")
        try:
            result = subprocess.run(['python', 'src/ml/train_model.py'], 
                                  capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                print("   ✅ ML model trained successfully")
            else:
                print(f"   ⚠️ ML training had issues: {result.stderr[:100]}...")
        except Exception as e:
            print(f"   ❌ Could not train ML model: {e}")
    
    # 3. Prepare ML data
    if Path('data/OnlineRetail.csv').exists():
        print("\n3️⃣ Preparing ML data...")
        try:
            result = subprocess.run(['python', 'src/ml/data_preparation.py'], 
                                  capture_output=True, text=True, timeout=90)
            if result.returncode == 0:
                print("   ✅ ML data preparation completed")
            else:
                print(f"   ⚠️ ML data prep had issues: {result.stderr[:100]}...")
        except Exception as e:
            print(f"   ❌ Could not prepare ML data: {e}")

def print_next_steps():
    """Print next steps for the user"""
    print("\n🎯 NEXT STEPS:")
    print("="*40)
    
    print("\n1. 🧪 Test your system:")
    print("   python test_complete_system.py")
    
    print("\n2. 🚀 Start the ML API:")
    print("   python src/ml/prediction_api.py")
    
    print("\n3. 📊 Run batch processing:")
    print("   python src/batch/daily_processor.py")
    
    print("\n4. 🖥️ Monitor your system:")
    print("   python src/monitoring/monitor.py")
    
    print("\n5. 🌊 Test streaming (optional):")
    print("   Terminal 1: python src/streaming/consumer.py")
    print("   Terminal 2: python src/streaming/publisher.py")
    
    print("\n6. 🧹 Clean data for team:")
    print("   python src/data_cleaning_utility.py")
    
    print("\n📚 For more details, check the README.md file!")

def main():
    """Main setup function"""
    print("🛠️ E-COMMERCE ANALYTICS PROJECT SETUP")
    print("="*50)
    print("This script will help you set up your project environment")
    
    # Step 1: Create directories
    create_directories()
    
    # Step 2: Check dependencies
    deps_ok = check_dependencies()
    
    # Step 3: Check data files
    data_ok = check_data_files()
    
    # Step 4: Run initial setup if possible
    if deps_ok and data_ok:
        print("\n✅ All dependencies and data files are ready!")
        
        response = input("\n❓ Run initial setup tasks (data analysis, ML training)? (y/n): ")
        if response.lower().strip() == 'y':
            run_initial_setup()
    
    # Step 5: Print next steps
    print_next_steps()
    
    # Final status
    if deps_ok and data_ok:
        print("\n🎉 Setup complete! Your project is ready to use.")
    else:
        print("\n⚠️ Setup incomplete. Please resolve the issues above.")

if __name__ == "__main__":
    main()
