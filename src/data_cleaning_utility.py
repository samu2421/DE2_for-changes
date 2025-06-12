# src/data_cleaning_utility.py
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_to_json_serializable(obj):
    """Convert numpy/pandas types to JSON serializable types"""
    import numpy as np
    import pandas as pd

    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    else:
        return obj


class EcommerceDataCleaner:
    """
    Comprehensive data cleaning utility for the Online Retail dataset
    Supports both Person B (ML training) and streaming data quality
    """
    
    def __init__(self, input_path='data/OnlineRetail.csv'):
        self.input_path = input_path
        self.df_raw = None
        self.df_cleaned = None
        self.cleaning_report = {}
        
    def load_raw_data(self):
        """Load the raw Kaggle dataset"""
        try:
            logger.info(f"Loading raw data from {self.input_path}")
            self.df_raw = pd.read_csv(self.input_path, encoding='latin1')
            logger.info(f"Raw data loaded: {len(self.df_raw):,} rows, {self.df_raw.shape[1]} columns")
            
            # Initial data quality assessment
            self.cleaning_report['initial'] = {
                'total_rows': len(self.df_raw),
                'total_columns': self.df_raw.shape[1],
                'memory_usage_mb': self.df_raw.memory_usage(deep=True).sum() / 1024**2,
                'missing_values': self.df_raw.isnull().sum().to_dict(),
                'duplicate_rows': self.df_raw.duplicated().sum()
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading raw data: {e}")
            return False
    
    def analyze_data_quality_issues(self):
        """Analyze and report data quality issues"""
        if self.df_raw is None:
            logger.error("No raw data loaded")
            return
        
        logger.info("Analyzing data quality issues...")
        
        issues = {
            'missing_values': {},
            'invalid_values': {},
            'data_type_issues': {},
            'outliers': {},
            'inconsistencies': {}
        }
        
        # Missing values analysis
        for col in self.df_raw.columns:
            missing_count = self.df_raw[col].isnull().sum()
            missing_pct = (missing_count / len(self.df_raw)) * 100
            if missing_count > 0:
                issues['missing_values'][col] = {
                    'count': int(missing_count),
                    'percentage': round(missing_pct, 2)
                }
        
        # Invalid values analysis
        if 'Quantity' in self.df_raw.columns:
            negative_qty = (self.df_raw['Quantity'] < 0).sum()
            zero_qty = (self.df_raw['Quantity'] == 0).sum()
            issues['invalid_values']['negative_quantity'] = int(negative_qty)
            issues['invalid_values']['zero_quantity'] = int(zero_qty)
        
        if 'UnitPrice' in self.df_raw.columns:
            negative_price = (self.df_raw['UnitPrice'] < 0).sum()
            zero_price = (self.df_raw['UnitPrice'] == 0).sum()
            issues['invalid_values']['negative_price'] = int(negative_price)
            issues['invalid_values']['zero_price'] = int(zero_price)
        
        # Date format issues
        if 'InvoiceDate' in self.df_raw.columns:
            try:
                date_series = pd.to_datetime(self.df_raw['InvoiceDate'], errors='coerce')
                invalid_dates = date_series.isnull().sum()
                issues['data_type_issues']['invalid_dates'] = int(invalid_dates)
            except:
                issues['data_type_issues']['date_parsing_error'] = True
        
        # Outlier detection
        numeric_columns = self.df_raw.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in ['Quantity', 'UnitPrice']:
                Q1 = self.df_raw[col].quantile(0.25)
                Q3 = self.df_raw[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_threshold = Q3 + 1.5 * IQR
                outliers = (self.df_raw[col] > outlier_threshold).sum()
                
                issues['outliers'][col] = {
                    'extreme_outliers': int(outliers),
                    'threshold': float(outlier_threshold),
                    'max_value': float(self.df_raw[col].max())
                }
        
        # Inconsistency checks
        if 'StockCode' in self.df_raw.columns:
            # Check for inconsistent product descriptions
            stock_desc = self.df_raw.groupby('StockCode')['Description'].nunique()
            inconsistent_products = (stock_desc > 1).sum()
            issues['inconsistencies']['products_with_multiple_descriptions'] = int(inconsistent_products)
        
        self.cleaning_report['quality_issues'] = issues
        
        # Print summary
        print("\n🔍 DATA QUALITY ISSUES SUMMARY:")
        print("=" * 40)
        
        if issues['missing_values']:
            print(f"📊 Missing Values:")
            for col, info in issues['missing_values'].items():
                print(f"  • {col}: {info['count']:,} ({info['percentage']:.1f}%)")
        
        if any([issues['invalid_values'][key] for key in issues['invalid_values'] if isinstance(issues['invalid_values'][key], int)]):
            print(f"\n⚠️  Invalid Values:")
            for key, count in issues['invalid_values'].items():
                if isinstance(count, int) and count > 0:
                    print(f"  • {key.replace('_', ' ').title()}: {count:,}")
        
        if issues['outliers']:
            print(f"\n📈 Outliers:")
            for col, info in issues['outliers'].items():
                print(f"  • {col}: {info['extreme_outliers']:,} extreme values")
    
    def clean_data_for_ml(self, aggressive_cleaning=False):
        """
        Clean data specifically for ML training
        
        Args:
            aggressive_cleaning: If True, applies more strict cleaning rules
        """
        if self.df_raw is None:
            logger.error("No raw data loaded")
            return None
        
        logger.info("Starting ML-focused data cleaning...")
        
        # Start with raw data
        df = self.df_raw.copy()
        initial_rows = len(df)
        
        cleaning_steps = []
        
        # Step 1: Remove rows with critical missing values
        critical_cols = ['CustomerID', 'InvoiceDate', 'Quantity', 'UnitPrice']
        missing_critical = df[critical_cols].isnull().any(axis=1).sum()
        df = df.dropna(subset=critical_cols)
        cleaning_steps.append(f"Removed {missing_critical:,} rows with missing critical values")
        
        # Step 2: Remove negative quantities and prices
        negative_qty = (df['Quantity'] <= 0).sum()
        df = df[df['Quantity'] > 0]
        cleaning_steps.append(f"Removed {negative_qty:,} rows with negative/zero quantities")
        
        negative_price = (df['UnitPrice'] <= 0).sum()
        df = df[df['UnitPrice'] > 0]
        cleaning_steps.append(f"Removed {negative_price:,} rows with negative/zero prices")
        
        # Step 3: Handle date formatting
        try:
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
            invalid_dates = df['InvoiceDate'].isnull().sum()
            if invalid_dates > 0:
                df = df.dropna(subset=['InvoiceDate'])
                cleaning_steps.append(f"Removed {invalid_dates:,} rows with invalid dates")
        except Exception as e:
            logger.warning(f"Date parsing issue: {e}")
        
        # Step 4: Remove extreme outliers (if aggressive cleaning)
        if aggressive_cleaning:
            # Quantity outliers
            qty_99 = df['Quantity'].quantile(0.99)
            qty_outliers = (df['Quantity'] > qty_99).sum()
            df = df[df['Quantity'] <= qty_99]
            cleaning_steps.append(f"Removed {qty_outliers:,} quantity outliers (>{qty_99:.0f})")
            
            # Price outliers
            price_99 = df['UnitPrice'].quantile(0.99)
            price_outliers = (df['UnitPrice'] > price_99).sum()
            df = df[df['UnitPrice'] <= price_99]
            cleaning_steps.append(f"Removed {price_outliers:,} price outliers (>£{price_99:.2f})")
        
        # Step 5: Clean text fields
        if 'Description' in df.columns:
            # Fill missing descriptions
            missing_desc = df['Description'].isnull().sum()
            df['Description'] = df['Description'].fillna('Unknown Product')
            if missing_desc > 0:
                cleaning_steps.append(f"Filled {missing_desc:,} missing product descriptions")
            
            # Clean description text
            df['Description'] = df['Description'].str.strip().str.upper()
        
        # Step 6: Standardize country names
        if 'Country' in df.columns:
            # Common country name standardization
            country_mapping = {
                'UNITED KINGDOM': 'United Kingdom',
                'UK': 'United Kingdom',
                'USA': 'United States',
                'US': 'United States'
            }
            
            df['Country'] = df['Country'].str.strip()
            df['Country'] = df['Country'].replace(country_mapping)
        
        # Step 7: Create calculated fields for ML
        df['Revenue'] = df['Quantity'] * df['UnitPrice']
        
        # Add time-based features
        df['Hour'] = df['InvoiceDate'].dt.hour
        df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
        df['Month'] = df['InvoiceDate'].dt.month
        df['Year'] = df['InvoiceDate'].dt.year
        df['Date'] = df['InvoiceDate'].dt.date
        
        # Final data validation
        final_rows = len(df)
        rows_removed = initial_rows - final_rows
        retention_rate = (final_rows / initial_rows) * 100
        
        self.df_cleaned = df
        
        # Update cleaning report
        self.cleaning_report['ml_cleaning'] = {
            'initial_rows': initial_rows,
            'final_rows': final_rows,
            'rows_removed': rows_removed,
            'retention_rate': round(retention_rate, 2),
            'cleaning_steps': cleaning_steps,
            'aggressive_cleaning': aggressive_cleaning
        }
        
        logger.info(f"ML cleaning complete: {final_rows:,} rows retained ({retention_rate:.1f}%)")
        
        return df
    
    def clean_data_for_streaming(self):
        """Clean data optimized for streaming use cases"""
        if self.df_raw is None:
            logger.error("No raw data loaded")
            return None
        
        logger.info("Starting streaming-focused data cleaning...")
        
        # Light cleaning for streaming (preserve more data)
        df = self.df_raw.copy()
        initial_rows = len(df)
        
        # Only remove critically invalid data
        df = df.dropna(subset=['Quantity', 'UnitPrice'])
        df = df[df['Quantity'] > 0]
        df = df[df['UnitPrice'] > 0]
        
        # Fill missing values instead of removing
        df['CustomerID'] = df['CustomerID'].fillna(99999)  # Anonymous customer
        df['Description'] = df['Description'].fillna('Unknown Product')
        df['Country'] = df['Country'].fillna('Unknown')
        
        # Basic date handling
        try:
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
            df = df.dropna(subset=['InvoiceDate'])
        except:
            logger.warning("Could not parse dates for streaming data")
        
        final_rows = len(df)
        retention_rate = (final_rows / initial_rows) * 100
        
        self.cleaning_report['streaming_cleaning'] = {
            'initial_rows': initial_rows,
            'final_rows': final_rows,
            'retention_rate': round(retention_rate, 2)
        }
        
        logger.info(f"Streaming cleaning complete: {final_rows:,} rows retained ({retention_rate:.1f}%)")
        
        return df
    
    def save_cleaned_data(self, output_dir='data/cleaned', include_both=True):
        """Save cleaned datasets"""
        Path(output_dir).mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if self.df_cleaned is not None:
            # Save ML-ready dataset
            ml_file = f"{output_dir}/ml_ready_data_{timestamp}.csv"
            self.df_cleaned.to_csv(ml_file, index=False)
            logger.info(f"ML-ready data saved to {ml_file}")
            
            # Save sample for quick testing
            sample_file = f"{output_dir}/ml_sample_1000.csv"
            self.df_cleaned.sample(min(1000, len(self.df_cleaned))).to_csv(sample_file, index=False)
            logger.info(f"ML sample data saved to {sample_file}")
        
        if include_both:
            # Save streaming-optimized dataset
            streaming_df = self.clean_data_for_streaming()
            if streaming_df is not None:
                streaming_file = f"{output_dir}/streaming_ready_data_{timestamp}.csv"
                streaming_df.to_csv(streaming_file, index=False)
                logger.info(f"Streaming-ready data saved to {streaming_file}")
    
    def generate_cleaning_report(self, save_to_file=True):
        """Generate comprehensive cleaning report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': self.cleaning_report,
            'recommendations': {
                'for_ml_training': [
                    "Use aggressive cleaning for better model performance",
                    "Consider feature engineering on temporal columns",
                    "Monitor for concept drift in new data",
                    "Validate data quality before each training run"
                ],
                'for_streaming': [
                    "Use light cleaning to preserve data volume",
                    "Implement real-time data validation",
                    "Monitor for data quality degradation",
                    "Set up alerts for unusual patterns"
                ],
                'for_batch_processing': [
                    "Run data quality checks before processing",
                    "Archive raw data before cleaning",
                    "Document cleaning decisions",
                    "Monitor cleaning effectiveness over time"
                ]
            }
        }
        
        if save_to_file:
            report_file = 'docs/data_cleaning_report.json'
            Path('docs').mkdir(exist_ok=True)
            with open(report_file, 'w') as f:
                serializable_report = convert_to_json_serializable(report)
                json.dump(serializable_report, f, indent=2)
            logger.info(f"Cleaning report saved to {report_file}")
        
        return report
    
    def print_summary(self):
        """Print a summary of cleaning results"""
        print("\n📋 DATA CLEANING SUMMARY")
        print("=" * 50)
        
        if 'initial' in self.cleaning_report:
            initial = self.cleaning_report['initial']
            print(f"📊 Initial Dataset:")
            print(f"  • Rows: {initial['total_rows']:,}")
            print(f"  • Columns: {initial['total_columns']}")
            print(f"  • Memory: {initial['memory_usage_mb']:.1f} MB")
            print(f"  • Duplicates: {initial['duplicate_rows']:,}")
        
        if 'ml_cleaning' in self.cleaning_report:
            ml = self.cleaning_report['ml_cleaning']
            print(f"\n🤖 ML-Ready Dataset:")
            print(f"  • Final Rows: {ml['final_rows']:,}")
            print(f"  • Retention Rate: {ml['retention_rate']:.1f}%")
            print(f"  • Rows Removed: {ml['rows_removed']:,}")
            print(f"  • Aggressive Cleaning: {'Yes' if ml['aggressive_cleaning'] else 'No'}")
        
        if 'streaming_cleaning' in self.cleaning_report:
            stream = self.cleaning_report['streaming_cleaning']
            print(f"\n🌊 Streaming-Ready Dataset:")
            print(f"  • Final Rows: {stream['final_rows']:,}")
            print(f"  • Retention Rate: {stream['retention_rate']:.1f}%")
        
        print(f"\n✅ Data cleaning complete!")
        print(f"📁 Check data/cleaned/ for output files")

def main():
    """Main data cleaning workflow"""
    print("🧹 E-COMMERCE DATA CLEANING UTILITY")
    print("=" * 50)
    print("This utility helps Person B (ML) and the team with clean, ready-to-use data")
    
    # Initialize cleaner
    cleaner = EcommerceDataCleaner()
    
    # Load raw data
    if not cleaner.load_raw_data():
        print("❌ Failed to load raw data. Make sure OnlineRetail.csv is in data/ folder")
        return
    
    # Analyze data quality
    cleaner.analyze_data_quality_issues()
    
    # Ask user for cleaning preferences
    print(f"\n❓ Choose cleaning level:")
    print("1. Light cleaning (for streaming/general use)")
    print("2. Aggressive cleaning (for ML training)")
    print("3. Both (recommended)")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        cleaner.clean_data_for_streaming()
    elif choice == "2":
        cleaner.clean_data_for_ml(aggressive_cleaning=True)
    elif choice == "3":
        cleaner.clean_data_for_ml(aggressive_cleaning=True)
    else:
        print("Invalid choice, using default (both)")
        cleaner.clean_data_for_ml(aggressive_cleaning=True)
    
    # Save cleaned data
    cleaner.save_cleaned_data()
    
    # Generate report
    cleaner.generate_cleaning_report()
    
    # Print summary
    cleaner.print_summary()

if __name__ == "__main__":
    main()