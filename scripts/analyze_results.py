#!/usr/bin/env python3
"""
Analysis script to understand successful vs failed frequency selection cases.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import *

def analyze_results():
    """Analyze the frequency selection results."""
    results_path = os.path.join(OUTPUT_DIR, "frequency_selection_results", "detailed_results.json")
    
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print("=== Frequency Selection Analysis ===")
    print(f"Total patients: {results['total_patients']}")
    print(f"Overall accuracy: {results['accuracy']:.1f}%")
    print(f"Tolerance: ±{results['tolerance_frames']} frames\n")
    
    # Categorize results
    successful_cases = []
    failed_cases = []
    
    for result in results['patient_results']:
        if result['is_correct']:
            successful_cases.append(result)
        else:
            failed_cases.append(result)
    
    print(f"Successful cases: {len(successful_cases)}")
    print(f"Failed cases: {len(failed_cases)}\n")
    
    # Analyze failure patterns
    analyze_failure_patterns(failed_cases)
    
    # Analyze frequency distributions
    analyze_frequency_distributions(results['patient_results'])
    
    # Analyze score characteristics
    analyze_score_characteristics(results['patient_results'])

def analyze_failure_patterns(failed_cases):
    """Analyze patterns in failed cases."""
    print("=== Failure Pattern Analysis ===")
    
    differences = [case['difference'] for case in failed_cases]
    
    print(f"Average prediction error: {np.mean(differences):.1f} frames")
    print(f"Max prediction error: {max(differences)} frames")
    print(f"Min prediction error: {min(differences)} frames")
    
    # Direction of errors
    direction_errors = defaultdict(int)
    for result in failed_cases:
        predicted = result['predicted_index']
        ground_truth = result['ground_truth_index']
        
        if predicted > ground_truth:
            direction_errors['over_predicted'] += 1
        else:
            direction_errors['under_predicted'] += 1
    
    print(f"Over-predicted (selected higher index): {direction_errors['over_predicted']}")
    print(f"Under-predicted (selected lower index): {direction_errors['under_predicted']}")
    
    # Analyze by frequency range
    extreme_selections = 0
    for result in failed_cases:
        predicted = result['predicted_index']
        series_length = result['series_length']
        
        if predicted in [0, series_length-1]:
            extreme_selections += 1
            print(f"  {result['patient_id']}: selected extreme frequency (index {predicted})")
    
    print(f"Extreme frequency selections: {extreme_selections}/{len(failed_cases)}")
    print()

def analyze_frequency_distributions(patient_results):
    """Analyze frequency distributions."""
    print("=== Frequency Distribution Analysis ===")
    
    predicted_frequencies = []
    ground_truth_frequencies = []
    
    for result in patient_results:
        predicted_frequencies.append(result['predicted_frequency'])
        ground_truth_frequencies.append(result['ground_truth_frequency'])
    
    print(f"Ground truth frequency range: {min(ground_truth_frequencies)} to {max(ground_truth_frequencies)} Hz")
    print(f"Predicted frequency range: {min(predicted_frequencies)} to {max(predicted_frequencies)} Hz")
    
    # Count frequency selections
    frequency_bins = defaultdict(int)
    for freq in predicted_frequencies:
        frequency_bins[freq] += 1
    
    print("Predicted frequency distribution:")
    for freq in sorted(frequency_bins.keys()):
        print(f"  {freq:4d} Hz: {frequency_bins[freq]} patients")
    
    print()

def analyze_score_characteristics(patient_results):
    """Analyze scoring characteristics."""
    print("=== Score Characteristics Analysis ===")
    
    score_ranges = []
    max_scores = []
    
    for result in patient_results:
        if 'weighting_scores' in result:
            scores = result['weighting_scores']
            score_range = max(scores) - min(scores)
            max_score = max(scores)
            
            score_ranges.append(score_range)
            max_scores.append(max_score)
    
    print(f"Average score range: {np.mean(score_ranges):.3f}")
    print(f"Average max score: {np.mean(max_scores):.3f}")
    print(f"Min score range: {min(score_ranges):.3f}")
    print(f"Max score range: {max(score_ranges):.3f}")
    
    # Identify cases with very small score ranges (poor discrimination)
    low_discrimination_cases = []
    for result in patient_results:
        if 'weighting_scores' in result:
            scores = result['weighting_scores']
            score_range = max(scores) - min(scores)
            
            if score_range < 0.1:  # Threshold for poor discrimination
                low_discrimination_cases.append((result['patient_id'], score_range))
    
    print(f"\nLow discrimination cases (score range < 0.1): {len(low_discrimination_cases)}")
    for patient_id, score_range in low_discrimination_cases:
        # Find the corresponding result
        patient_result = None
        for result in patient_results:
            if result['patient_id'] == patient_id:
                patient_result = result
                break
        
        is_correct = patient_result['is_correct'] if patient_result else False
        print(f"  {patient_id}: range={score_range:.3f}, correct={is_correct}")
    
    print()

def index_to_frequency(index, freq_step, series_length):
    """Convert image index to frequency offset."""
    center_index = series_length // 2
    frequency_offset = (index - center_index) * freq_step
    return frequency_offset

def main():
    """Run analysis."""
    analyze_results()

if __name__ == "__main__":
    main()