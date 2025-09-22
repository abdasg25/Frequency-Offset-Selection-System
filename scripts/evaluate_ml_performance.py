#!/usr/bin/env python3
"""
ML-based frequency selection runner and evaluation script.
"""

import os
import sys
import argparse
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_ml_based_approach(model_path=None):
    """Run the ML-based frequency selection."""
    print("🤖 Running ML-Based Frequency Selection...")
    
    # Check if model exists
    if model_path is None:
        model_path = "outputs/ml_frequency_models/best_model.pth"
    
    if not os.path.exists(model_path):
        print(f"ML model not found at {model_path}")
        print("Please train the ML model first:")
        print("python scripts/train_ml_frequency_selector.py")
        return None
    
    # Import and run ML system
    from scripts.run_ml_frequency_selection import run_ml_frequency_selection_system
    
    results = run_ml_frequency_selection_system(model_path=model_path)
    return results

def evaluate_ml_performance(ml_results):
    """Evaluate ML performance and show detailed analysis."""
    if ml_results is None:
        print("Cannot evaluate - ML approach failed")
        return
    
    print("\n" + "="*60)
    print("📊 ML-BASED FREQUENCY SELECTION EVALUATION")
    print("="*60)
    
    # Overall accuracy
    ml_acc = ml_results['accuracy']
    target_accuracy = 92.0
    
    print(f"\n Performance Analysis:")
    print(f"   ML Accuracy: {ml_acc:.1f}%")
    print(f"   Target: {target_accuracy:.1f}%")
    
    if ml_acc >= target_accuracy:
        print("    TARGET ACHIEVED! 🎉")
    else:
        needed = target_accuracy - ml_acc
        print(f"   Need +{needed:.1f}% to reach target")
    
    # Patient results breakdown
    print(f"\n Patient Results:")
    print(f"   Correct: {ml_results['correct_predictions']}/{ml_results['total_patients']} patients")
    
    # Failed cases analysis
    failed_cases = [r for r in ml_results['patient_results'] if not r['is_correct']]
    if failed_cases:
        print(f"\nFailed Cases ({len(failed_cases)}):")
        for result in failed_cases:
            patient_id = result['patient_id']
            difference = result['difference']
            confidence = result.get('model_confidence', 'N/A')
            gt_freq = result['ground_truth_frequency']
            pred_freq = result['predicted_frequency']
            print(f"   {patient_id}: {gt_freq}Hz → {pred_freq}Hz (diff: {difference}, conf: {confidence})")
    
    # Confidence analysis
    if 'patient_results' in ml_results and ml_results['patient_results']:
        if 'model_confidence' in ml_results['patient_results'][0]:
            confidences = [r.get('model_confidence', 0) for r in ml_results['patient_results']]
            avg_confidence = sum(confidences) / len(confidences)
            
            high_conf_results = [r for r in ml_results['patient_results'] if r.get('model_confidence', 0) > 0.8]
            low_conf_results = [r for r in ml_results['patient_results'] if r.get('model_confidence', 0) < 0.5]
            
            print(f"\n Confidence Analysis:")
            print(f"   Average Confidence: {avg_confidence:.3f}")
            print(f"   High Confidence (>0.8): {len(high_conf_results)} patients")
            print(f"   Low Confidence (<0.5): {len(low_conf_results)} patients")
            
            if high_conf_results:
                high_conf_accuracy = sum(r['is_correct'] for r in high_conf_results) / len(high_conf_results) * 100
                print(f"   High Confidence Accuracy: {high_conf_accuracy:.1f}%")

def main():
    """Main function for ML evaluation."""
    parser = argparse.ArgumentParser(description='Run and Evaluate ML-Based Frequency Selection')
    parser.add_argument('--ml_model_path', type=str, default=None,
                       help='Path to trained ML model')
    parser.add_argument('--save_results', action='store_true',
                       help='Save evaluation results to file')
    
    args = parser.parse_args()
    
    print("� ML-Based Frequency Offset Selection Evaluation")
    print("=" * 50)
    
    # Run ML approach
    ml_results = run_ml_based_approach(args.ml_model_path)
    
    # Evaluate performance
    if ml_results:
        evaluate_ml_performance(ml_results)
        
        # Save results if requested
        if args.save_results:
            results_summary = {
                'ml_accuracy': ml_results['accuracy'],
                'target_achieved': ml_results['accuracy'] >= 92.0,
                'correct_predictions': ml_results['correct_predictions'],
                'total_patients': ml_results['total_patients'],
                'model_path': args.ml_model_path or 'outputs/ml_frequency_models/best_model.pth'
            }
            
            summary_path = 'outputs/ml_evaluation_summary.json'
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            with open(summary_path, 'w') as f:
                json.dump(results_summary, f, indent=2)
            
            print(f"\n💾 Evaluation summary saved to: {summary_path}")
    else:
        print(" ML evaluation failed. Please check model path and training.")

if __name__ == "__main__":
    main()