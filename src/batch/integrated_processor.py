# src/batch/integrated_processor.py
import pandas as pd
import numpy as np
import requests
import json
import pickle
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys
import os

# Import your existing batch processor
sys.path.append('.')
from batch.daily_processor import EcommerceBatchProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegratedMLBatchProcessor(EcommerceBatchProcessor):
    """
    Enhanced batch processor that integrates ML predictions
    Extends the base batch processor with ML capabilities
    """
    
    def __init__(self, data_path='data/OnlineRetail.csv', ml_api_url='http://localhost:5001'):
        super().__init__(data_path)
        self.ml_api_url = ml_api_url
        self.ml_predictions = []
        self.api_available = False
        self.check_ml_api()
    
    def check_ml_api(self):
        """Check if ML API is available"""
        try:
            response = requests.get(f"{self.ml_api_url}/health", timeout=5)
            if response.status_code == 200:
                self.api_available = True
                logger.info("✅ ML API is available and healthy")
            else:
                logger.warning(f"⚠️ ML API responded with status {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️ ML API not available: {e}")
            logger.info("Batch processing will continue without ML predictions")
    
    def get_ml_prediction(self, order_data):
        """Get ML prediction for a single order"""
        if not self.api_available:
            return None
            
        try:
            # Prepare prediction request matching your API format
            prediction_request = {
                'quantity': int(order_data.get('Quantity', 1)),
                'unit_price': float(order_data.get('UnitPrice', 0)),
                'hour': int(order_data.get('Hour', 12)),
                'day_of_week': int(order_data.get('DayOfWeek', 1))
            }
            
            # Make API request
            response = requests.post(
                f"{self.ml_api_url}/predict",
                json=prediction_request,
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('predicted_revenue', 0)
            else:
                logger.warning(f"ML API returned status {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"ML API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting ML prediction: {e}")
            return None
    
    def process_daily_metrics_with_ml(self, target_date=None, sample_size=100):
        """
        Enhanced daily processing with ML predictions
        Uses sample_size to limit API calls for demonstration
        """
        # First, run the standard daily processing
        metrics = super().process_daily_metrics(target_date)
        if not metrics:
            return None
        
        logger.info("Adding ML predictions to daily metrics...")
        
        # Get the target date data
        if target_date is None:
            target_date = self.df['Date'].max()
        
        daily_data = self.df[self.df['Date'] == target_date].copy()
        
        # Sample data for ML predictions (to avoid overwhelming the API)
        if len(daily_data) > sample_size:
            daily_sample = daily_data.sample(n=sample_size, random_state=42)
            logger.info(f"Sampling {sample_size} orders from {len(daily_data)} for ML predictions")
        else:
            daily_sample = daily_data
        
        # Add ML predictions for each order in sample
        predictions = []
        successful_predictions = 0
        
        for _, row in daily_sample.iterrows():
            order_data = {
                'Quantity': row['Quantity'],
                'UnitPrice': row['UnitPrice'],
                'Hour': row['Hour'],
                'DayOfWeek': row['DayOfWeek']
            }
            
            predicted_revenue = self.get_ml_prediction(order_data)
            actual_revenue = row['Revenue']
            
            prediction_record = {
                'invoice_no': row['InvoiceNo'],
                'actual_revenue': round(actual_revenue, 2),
                'predicted_revenue': round(predicted_revenue, 2) if predicted_revenue else None,
                'prediction_error': round(abs(actual_revenue - predicted_revenue), 2) if predicted_revenue else None,
                'prediction_accuracy': round((1 - abs(actual_revenue - predicted_revenue) / actual_revenue) * 100, 2) if predicted_revenue and actual_revenue > 0 else None
            }
            
            predictions.append(prediction_record)
            
            if predicted_revenue is not None:
                successful_predictions += 1
        
        # Calculate ML metrics
        valid_predictions = [p for p in predictions if p['predicted_revenue'] is not None]
        
        if valid_predictions:
            ml_metrics = {
                'total_predictions_attempted': len(predictions),
                'successful_predictions': successful_predictions,
                'prediction_success_rate': round(successful_predictions / len(predictions) * 100, 2),
                'average_prediction_error': round(np.mean([p['prediction_error'] for p in valid_predictions]), 2),
                'average_prediction_accuracy': round(np.mean([p['prediction_accuracy'] for p in valid_predictions if p['prediction_accuracy'] is not None]), 2),
                'total_predicted_revenue': round(sum([p['predicted_revenue'] for p in valid_predictions]), 2),
                'total_actual_revenue': round(sum([p['actual_revenue'] for p in valid_predictions]), 2)
            }
            
            # Add ML metrics to the daily metrics
            metrics['ml_predictions'] = ml_metrics
            metrics['prediction_sample_size'] = len(predictions)
            metrics['prediction_details'] = predictions[:10]  # Save first 10 for review
            
            logger.info(f"ML predictions added: {successful_predictions}/{len(predictions)} successful")
            logger.info(f"Average prediction accuracy: {ml_metrics['average_prediction_accuracy']:.1f}%")
        
        else:
            logger.warning("No successful ML predictions obtained")
            metrics['ml_predictions'] = {'error': 'No successful predictions', 'api_available': self.api_available}
        
        self.processed_metrics = metrics
        return metrics
    
    def generate_revenue_forecast(self, forecast_days=7):
        """
        Generate revenue forecast using historical patterns and ML predictions
        """
        logger.info(f"Generating {forecast_days}-day revenue forecast...")
        
        if self.df is None:
            logger.error("Data not loaded")
            return None
        
        # Calculate historical daily patterns
        historical_patterns = self.df.groupby(['DayOfWeek', 'Hour']).agg({
            'Revenue': 'mean',
            'InvoiceNo': 'count',
            'Quantity': 'mean',
            'UnitPrice': 'mean'
        }).reset_index()
        
        forecasts = []
        current_date = datetime.now().date()
        
        for day_offset in range(forecast_days):
            forecast_date = current_date + timedelta(days=day_offset)
            day_of_week = forecast_date.weekday()
            
            # Get patterns for this day of week
            day_patterns = historical_patterns[historical_patterns['DayOfWeek'] == day_of_week]
            
            if not day_patterns.empty:
                daily_forecast = {
                    'date': str(forecast_date),
                    'day_of_week': day_of_week,
                    'day_name': forecast_date.strftime('%A'),
                    'historical_revenue': 0,
                    'ml_predicted_revenue': 0,
                    'confidence': 'medium'
                }
                
                # Calculate historical baseline
                historical_revenue = day_patterns['Revenue'].sum()
                daily_forecast['historical_revenue'] = round(historical_revenue, 2)
                
                # Get ML prediction for representative order
                if self.api_available:
                    sample_order = {
                        'Quantity': int(day_patterns['Quantity'].mean()),
                        'UnitPrice': day_patterns['UnitPrice'].mean(),
                        'Hour': 12,  # Noon as representative hour
                        'DayOfWeek': day_of_week
                    }
                    
                    ml_prediction = self.get_ml_prediction(sample_order)
                    if ml_prediction:
                        # Scale up prediction by number of expected orders
                        expected_orders = day_patterns['InvoiceNo'].sum()
                        daily_forecast['ml_predicted_revenue'] = round(ml_prediction * expected_orders / 100, 2)  # Scale down for realism
                        daily_forecast['confidence'] = 'high'
                
                forecasts.append(daily_forecast)
        
        logger.info(f"Forecast generated for {len(forecasts)} days")
        return forecasts
    
    def analyze_prediction_performance(self):
        """Analyze ML prediction performance across different segments"""
        if not self.processed_metrics or 'ml_predictions' not in self.processed_metrics:
            logger.warning("No ML predictions available for performance analysis")
            return None
        
        predictions = self.processed_metrics.get('prediction_details', [])
        valid_predictions = [p for p in predictions if p['predicted_revenue'] is not None]
        
        if not valid_predictions:
            return None
        
        # Performance by revenue range
        performance_analysis = {
            'total_analyzed': len(valid_predictions),
            'overall_accuracy': round(np.mean([p['prediction_accuracy'] for p in valid_predictions if p['prediction_accuracy'] is not None]), 2),
            'revenue_ranges': {}
        }
        
        # Categorize by revenue ranges
        for pred in valid_predictions:
            actual = pred['actual_revenue']
            if actual <= 10:
                category = 'low (≤£10)'
            elif actual <= 50:
                category = 'medium (£10-50)'
            elif actual <= 100:
                category = 'high (£50-100)'
            else:
                category = 'very_high (>£100)'
            
            if category not in performance_analysis['revenue_ranges']:
                performance_analysis['revenue_ranges'][category] = []
            
            performance_analysis['revenue_ranges'][category].append(pred['prediction_accuracy'])
        
        # Calculate averages for each range
        for category, accuracies in performance_analysis['revenue_ranges'].items():
            valid_accuracies = [acc for acc in accuracies if acc is not None]
            if valid_accuracies:
                performance_analysis['revenue_ranges'][category] = {
                    'count': len(valid_accuracies),
                    'avg_accuracy': round(np.mean(valid_accuracies), 2),
                    'min_accuracy': round(min(valid_accuracies), 2),
                    'max_accuracy': round(max(valid_accuracies), 2)
                }
        
        return performance_analysis
    
    def generate_integrated_report(self):
        """Generate comprehensive report with ML insights"""
        if not self.processed_metrics:
            logger.error("No processed metrics available")
            return
        
        # Generate base report
        super().generate_daily_report()
        
        # Add ML-specific insights
        if 'ml_predictions' in self.processed_metrics:
            ml_metrics = self.processed_metrics['ml_predictions']
            
            print("🤖 ML PREDICTION INSIGHTS:")
            print("=" * 50)
            
            if 'error' not in ml_metrics:
                print(f"📊 Prediction Success Rate: {ml_metrics['prediction_success_rate']}%")
                print(f"🎯 Average Prediction Accuracy: {ml_metrics['average_prediction_accuracy']}%")
                print(f"💰 Predicted vs Actual Revenue: £{ml_metrics['total_predicted_revenue']:,} vs £{ml_metrics['total_actual_revenue']:,}")
                
                revenue_diff = ml_metrics['total_predicted_revenue'] - ml_metrics['total_actual_revenue']
                revenue_diff_pct = (revenue_diff / ml_metrics['total_actual_revenue']) * 100 if ml_metrics['total_actual_revenue'] > 0 else 0
                print(f"📈 Revenue Prediction Difference: £{revenue_diff:+,.2f} ({revenue_diff_pct:+.1f}%)")
                
                print(f"\n🔍 Sample Predictions:")
                for i, pred in enumerate(self.processed_metrics.get('prediction_details', [])[:5], 1):
                    if pred['predicted_revenue']:
                        accuracy = pred.get('prediction_accuracy', 0) or 0
                        print(f"  {i}. Invoice {pred['invoice_no']}: £{pred['actual_revenue']} actual → £{pred['predicted_revenue']} predicted ({accuracy:.1f}% accuracy)")
                
                # Performance analysis
                perf_analysis = self.analyze_prediction_performance()
                if perf_analysis:
                    print(f"\n📈 Performance by Revenue Range:")
                    for range_name, stats in perf_analysis['revenue_ranges'].items():
                        if isinstance(stats, dict):
                            print(f"  • {range_name}: {stats['avg_accuracy']:.1f}% accuracy ({stats['count']} orders)")
            else:
                print("❌ ML predictions not available")
                if not ml_metrics.get('api_available', False):
                    print("💡 Start the ML API with: python src/ml/prediction_api.py")
        
        print("=" * 50 + "\n")
    
    def save_integrated_results(self, output_dir='data/processed'):
        """Save integrated results including ML predictions"""
        # Call parent save method
        super().save_results(output_dir)
        
        # Save additional ML-specific results
        if hasattr(self, 'processed_metrics') and self.processed_metrics:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save ML prediction results
            if 'ml_predictions' in self.processed_metrics:
                ml_file = f"{output_dir}/ml_predictions_{timestamp}.json"
                with open(ml_file, 'w') as f:
                    json.dump(self.processed_metrics['ml_predictions'], f, indent=2)
                logger.info(f"ML predictions saved to {ml_file}")
            
            # Save forecasts
            forecasts = self.generate_revenue_forecast()
            if forecasts:
                forecast_file = f"{output_dir}/revenue_forecast_{timestamp}.json"
                with open(forecast_file, 'w') as f:
                    json.dump(forecasts, f, indent=2)
                logger.info(f"Revenue forecast saved to {forecast_file}")

def main():
    """Main integrated batch processing function"""
    print("🚀 Starting Integrated ML Batch Processing...")
    print("=" * 60)
    
    # Initialize integrated processor
    processor = IntegratedMLBatchProcessor()
    
    # Load data
    if not processor.load_data():
        print("❌ Failed to load data. Exiting.")
        return
    
    # Process daily metrics with ML
    print("\n📊 Processing daily metrics with ML integration...")
    metrics = processor.process_daily_metrics_with_ml(sample_size=50)  # Use 50 orders for demo
    
    if metrics:
        print("\n📋 Generating integrated report...")
        processor.generate_integrated_report()
    
    # Generate forecasts 
    print("\n📈 Generating revenue forecasts...")
    forecasts = processor.generate_revenue_forecast(forecast_days=5)
    if forecasts:
        print("\n🔮 5-DAY REVENUE FORECAST:")
        for forecast in forecasts:
            hist_rev = forecast['historical_revenue']
            ml_rev = forecast['ml_predicted_revenue']
            confidence = forecast['confidence']
            print(f"  📅 {forecast['date']} ({forecast['day_name']}): £{hist_rev:,.2f} historical | £{ml_rev:,.2f} ML predicted | {confidence} confidence")
    
    # Save all results
    print("\n💾 Saving integrated results...")
    processor.save_integrated_results()
    
    print("\n✅ Integrated batch processing completed successfully!")
    print("📁 Check data/processed/ for all generated files")

if __name__ == "__main__":
    main()
