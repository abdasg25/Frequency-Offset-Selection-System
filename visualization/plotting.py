import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import cv2

class FrequencyVisualizationManager:
    """
    Comprehensive visualization manager for frequency offset selection results.
    
    This class creates various plots and visualizations to analyze and present
    the results of the automated frequency offset selection system.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the visualization manager.
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_frequency_selection_results(
        self,
        patient_id: str,
        frequency_series: np.ndarray,
        heart_mask: np.ndarray,
        analysis_results: Dict,
        ground_truth_idx: int,
        predicted_idx: int
    ):
        """
        Create comprehensive visualization for frequency selection results.
        
        Args:
            patient_id: Patient identifier
            frequency_series: Complete frequency series
            heart_mask: Heart ROI mask
            analysis_results: Results from frequency analysis
            ground_truth_idx: Ground truth optimal frequency index
            predicted_idx: Predicted optimal frequency index
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Original frequency series (selected frames)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_frequency_series_overview(
            ax1, frequency_series, ground_truth_idx, predicted_idx
        )
        
        # 2. Heart mask overlay
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_heart_mask_overlay(ax2, frequency_series, heart_mask, predicted_idx)
        
        # 3. Weighting scores plot
        ax3 = fig.add_subplot(gs[0, 3])
        self._plot_weighting_scores(
            ax3, analysis_results['weighting_scores'], ground_truth_idx, predicted_idx
        )
        
        # 4. High-frequency energy plot
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_high_frequency_energy(
            ax4, analysis_results['high_frequency_energies'], 
            analysis_results['selected_low_freq_indices']
        )
        
        # 5. Weighting maps for key frames
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])
        ax7 = fig.add_subplot(gs[1, 3])
        self._plot_weighting_maps(
            [ax5, ax6, ax7], analysis_results['weighting_maps'],
            [ground_truth_idx, predicted_idx, analysis_results['selected_low_freq_indices'][0]]
        )
        
        # 6. Reference median and comparison
        ax8 = fig.add_subplot(gs[2, 0])
        ax9 = fig.add_subplot(gs[2, 1])
        self._plot_reference_median_comparison(
            ax8, ax9, analysis_results['reference_median'], 
            frequency_series[predicted_idx], heart_mask
        )
        
        # 7. High-frequency components visualization
        ax10 = fig.add_subplot(gs[2, 2])
        ax11 = fig.add_subplot(gs[2, 3])
        self._plot_high_frequency_components(
            ax10, ax11, analysis_results['high_freq_components'],
            ground_truth_idx, predicted_idx
        )
        
        # Add main title
        accuracy_status = "✓ CORRECT" if abs(predicted_idx - ground_truth_idx) <= 2 else "✗ INCORRECT"
        fig.suptitle(
            f"Frequency Offset Selection Results - {patient_id} ({accuracy_status})\n"
            f"Ground Truth: Index {ground_truth_idx} | Predicted: Index {predicted_idx}",
            fontsize=16, fontweight='bold'
        )
        
        # Save figure
        output_path = os.path.join(self.output_dir, f"{patient_id}_analysis.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved: {output_path}")
    
    def _plot_frequency_series_overview(
        self, ax, frequency_series: np.ndarray, gt_idx: int, pred_idx: int
    ):
        """Plot overview of frequency series with key frames highlighted."""
        n_frames = min(frequency_series.shape[0], 8)  # Show max 8 frames
        indices = np.linspace(0, frequency_series.shape[0]-1, n_frames, dtype=int)
        
        for i, idx in enumerate(indices):
            img = frequency_series[idx]
            
            # Create subplot
            ax_sub = plt.subplot2grid((2, 4), (i//4, i%4), fig=ax.figure)
            ax_sub.imshow(img, cmap='gray')
            ax_sub.set_title(f"Frame {idx}")
            ax_sub.axis('off')
            
            # Highlight ground truth and prediction
            if idx == gt_idx:
                ax_sub.add_patch(plt.Rectangle((0, 0), img.shape[1], img.shape[0], 
                                             fill=False, edgecolor='green', linewidth=3))
                ax_sub.set_title(f"Frame {idx} (GT)", color='green', fontweight='bold')
            elif idx == pred_idx:
                ax_sub.add_patch(plt.Rectangle((0, 0), img.shape[1], img.shape[0], 
                                             fill=False, edgecolor='red', linewidth=3))
                ax_sub.set_title(f"Frame {idx} (Pred)", color='red', fontweight='bold')
        
        ax.set_title("Frequency Series Overview", fontsize=12, fontweight='bold')
        ax.axis('off')
    
    def _plot_heart_mask_overlay(
        self, ax, frequency_series: np.ndarray, heart_mask: np.ndarray, pred_idx: int
    ):
        """Plot heart mask overlay on predicted optimal frame."""
        img = frequency_series[pred_idx]
        
        # Create overlay
        overlay = np.zeros((*img.shape, 3))
        overlay[:, :, 0] = img  # Red channel
        overlay[:, :, 1] = img  # Green channel
        overlay[:, :, 2] = img  # Blue channel
        
        # Add heart mask in red
        mask_overlay = heart_mask > 0.5
        overlay[mask_overlay, 0] = 1.0  # Red for heart
        overlay[mask_overlay, 1] = 0.0
        overlay[mask_overlay, 2] = 0.0
        
        ax.imshow(overlay)
        ax.set_title(f"Heart ROI Mask\n(Frame {pred_idx})", fontweight='bold')
        ax.axis('off')
    
    def _plot_weighting_scores(
        self, ax, weighting_scores: List[float], gt_idx: int, pred_idx: int
    ):
        """Plot weighting scores across frequency offsets."""
        x = np.arange(len(weighting_scores))
        ax.plot(x, weighting_scores, 'b-', linewidth=2, label='Weighting Score')
        ax.axvline(gt_idx, color='green', linestyle='--', linewidth=2, label='Ground Truth')
        ax.axvline(pred_idx, color='red', linestyle='--', linewidth=2, label='Predicted')
        
        ax.set_xlabel('Frequency Offset Index')
        ax.set_ylabel('Weighting Score')
        ax.set_title('Weighting Scores', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_high_frequency_energy(
        self, ax, energy_values: List[float], selected_indices: List[int]
    ):
        """Plot high-frequency energy across frequency offsets."""
        x = np.arange(len(energy_values))
        ax.plot(x, energy_values, 'purple', linewidth=2, label='HF Energy')
        
        # Highlight selected low-frequency images
        for idx in selected_indices:
            ax.axvline(idx, color='orange', linestyle=':', alpha=0.7)
        
        ax.set_xlabel('Frequency Offset Index')
        ax.set_ylabel('High-Frequency Energy')
        ax.set_title('High-Frequency Energy\n(Lower is Better)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_weighting_maps(
        self, axes: List, weighting_maps: np.ndarray, indices: List[int]
    ):
        """Plot weighting maps for specific frames."""
        titles = ['Ground Truth', 'Predicted', 'Low HF Energy']
        
        for i, (ax, idx) in enumerate(zip(axes, indices)):
            if idx < weighting_maps.shape[0]:
                wmap = weighting_maps[idx]
                im = ax.imshow(wmap, cmap='hot', vmin=0, vmax=1)
                ax.set_title(f'{titles[i]}\nWeighting Map (Frame {idx})', fontweight='bold')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.axis('off')
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_reference_median_comparison(
        self, ax1, ax2, reference_median: np.ndarray, 
        predicted_image: np.ndarray, heart_mask: np.ndarray
    ):
        """Plot reference median and comparison with predicted image."""
        # Reference median
        ax1.imshow(reference_median, cmap='gray')
        ax1.set_title('Reference Median\n(from Low HF Images)', fontweight='bold')
        ax1.axis('off')
        
        # Difference map
        diff = np.abs(predicted_image - reference_median) * heart_mask
        im = ax2.imshow(diff, cmap='jet')
        ax2.set_title('Predicted vs Median\nDifference (ROI)', fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    def _plot_high_frequency_components(
        self, ax1, ax2, high_freq_components: np.ndarray, gt_idx: int, pred_idx: int
    ):
        """Plot high-frequency components for ground truth and predicted frames."""
        # Ground truth high-frequency components
        if gt_idx < high_freq_components.shape[0]:
            hf_gt = high_freq_components[gt_idx]
            ax1.imshow(np.abs(hf_gt), cmap='viridis')
            ax1.set_title(f'HF Components (GT)\nFrame {gt_idx}', fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax1.transAxes)
        ax1.axis('off')
        
        # Predicted high-frequency components
        if pred_idx < high_freq_components.shape[0]:
            hf_pred = high_freq_components[pred_idx]
            ax2.imshow(np.abs(hf_pred), cmap='viridis')
            ax2.set_title(f'HF Components (Pred)\nFrame {pred_idx}', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax2.transAxes)
        ax2.axis('off')
    
    def plot_overall_performance(
        self, results: List[Dict], output_filename: str = "overall_performance.png"
    ):
        """
        Plot overall system performance across all patients.
        
        Args:
            results: List of patient results dictionaries
            output_filename: Output filename for the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Extract data
        differences = [r['difference'] for r in results]
        is_correct = [r['is_correct'] for r in results]
        gt_frequencies = [r['ground_truth_frequency'] for r in results]
        pred_frequencies = [r['predicted_frequency'] for r in results]
        
        # 1. Accuracy histogram
        ax = axes[0, 0]
        accuracy = sum(is_correct) / len(is_correct) * 100
        ax.bar(['Correct', 'Incorrect'], 
               [sum(is_correct), len(is_correct) - sum(is_correct)],
               color=['green', 'red'], alpha=0.7)
        ax.set_title(f'Overall Accuracy: {accuracy:.1f}%', fontweight='bold')
        ax.set_ylabel('Number of Cases')
        
        # 2. Difference distribution
        ax = axes[0, 1]
        ax.hist(differences, bins=range(max(differences)+2), alpha=0.7, color='blue')
        ax.set_xlabel('Prediction Error (frames)')
        ax.set_ylabel('Number of Cases')
        ax.set_title('Prediction Error Distribution', fontweight='bold')
        ax.axvline(2, color='red', linestyle='--', label='Tolerance (±2 frames)')
        ax.legend()
        
        # 3. Ground truth vs Predicted scatter
        ax = axes[0, 2]
        colors = ['green' if correct else 'red' for correct in is_correct]
        ax.scatter(gt_frequencies, pred_frequencies, c=colors, alpha=0.7)
        ax.plot([-150, 150], [-150, 150], 'k--', alpha=0.5, label='Perfect Prediction')
        ax.set_xlabel('Ground Truth Frequency (Hz)')
        ax.set_ylabel('Predicted Frequency (Hz)')
        ax.set_title('Predicted vs Ground Truth', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Error by ground truth frequency
        ax = axes[1, 0]
        unique_freqs = sorted(set(gt_frequencies))
        freq_errors = []
        for freq in unique_freqs:
            freq_diffs = [d for gt, d in zip(gt_frequencies, differences) if gt == freq]
            freq_errors.append(np.mean(freq_diffs) if freq_diffs else 0)
        
        ax.bar(unique_freqs, freq_errors, alpha=0.7, color='orange')
        ax.set_xlabel('Ground Truth Frequency (Hz)')
        ax.set_ylabel('Mean Prediction Error (frames)')
        ax.set_title('Error by Ground Truth Frequency', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        
        # 5. Cumulative accuracy
        ax = axes[1, 1]
        tolerances = range(6)  # 0 to 5 frames tolerance
        cumulative_acc = []
        for tol in tolerances:
            acc = sum(1 for d in differences if d <= tol) / len(differences) * 100
            cumulative_acc.append(acc)
        
        ax.plot(tolerances, cumulative_acc, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Tolerance (frames)')
        ax.set_ylabel('Cumulative Accuracy (%)')
        ax.set_title('Accuracy vs Tolerance', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(92.1, color='red', linestyle='--', label='Paper Accuracy (92.1%)')
        ax.legend()
        
        # 6. Patient-wise results
        ax = axes[1, 2]
        patient_ids = [r['patient_id'] for r in results]
        colors = ['green' if correct else 'red' for correct in is_correct]
        y_pos = np.arange(len(patient_ids))
        
        bars = ax.barh(y_pos, differences, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([pid.replace('Patient-', '') for pid in patient_ids])
        ax.set_xlabel('Prediction Error (frames)')
        ax.set_title('Error by Patient', fontweight='bold')
        ax.axvline(2, color='black', linestyle='--', alpha=0.7, label='Tolerance')
        ax.legend()
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Overall performance plot saved: {output_path}")
    
    def create_summary_report(
        self, results: List[Dict], output_filename: str = "summary_report.txt"
    ):
        """
        Create a text summary report of the system performance.
        
        Args:
            results: List of patient results dictionaries
            output_filename: Output filename for the report
        """
        # Calculate statistics
        total_patients = len(results)
        correct_predictions = sum(r['is_correct'] for r in results)
        accuracy = correct_predictions / total_patients * 100
        
        differences = [r['difference'] for r in results]
        mean_error = np.mean(differences)
        std_error = np.std(differences)
        max_error = max(differences)
        
        # Failed cases
        failed_cases = [r for r in results if not r['is_correct']]
        
        # Create report
        report = f"""
AUTOMATED FREQUENCY OFFSET SELECTION SYSTEM
PERFORMANCE SUMMARY REPORT
==========================================

OVERALL PERFORMANCE:
- Total Patients Processed: {total_patients}
- Correct Predictions: {correct_predictions}
- Overall Accuracy: {accuracy:.1f}%
- Tolerance: ±2 frames

ERROR STATISTICS:
- Mean Prediction Error: {mean_error:.2f} frames
- Standard Deviation: {std_error:.2f} frames
- Maximum Error: {max_error} frames

COMPARISON WITH PAPER:
- Paper Accuracy: 92.1%
- Our Accuracy: {accuracy:.1f}%
- Performance Difference: {accuracy - 92.1:+.1f}%

FAILED CASES ({len(failed_cases)} patients):
"""
        
        for case in failed_cases:
            report += f"- {case['patient_id']}: Error = {case['difference']} frames "
            report += f"(GT: {case['ground_truth_frequency']} Hz, "
            report += f"Pred: {case['predicted_frequency']} Hz)\n"
        
        report += f"""
FREQUENCY DISTRIBUTION:
Ground Truth Frequencies (Hz): {sorted(set(r['ground_truth_frequency'] for r in results))}

SYSTEM ROBUSTNESS:
- Cases within ±1 frame: {sum(1 for d in differences if d <= 1)}/{total_patients} ({sum(1 for d in differences if d <= 1)/total_patients*100:.1f}%)
- Cases within ±2 frames: {sum(1 for d in differences if d <= 2)}/{total_patients} ({sum(1 for d in differences if d <= 2)/total_patients*100:.1f}%)
- Cases within ±3 frames: {sum(1 for d in differences if d <= 3)}/{total_patients} ({sum(1 for d in differences if d <= 3)/total_patients*100:.1f}%)

RECOMMENDATIONS:
"""
        
        if accuracy >= 90:
            report += "✓ System performance is excellent and meets clinical requirements.\n"
        elif accuracy >= 80:
            report += "⚠ System performance is good but may benefit from additional training data.\n"
        else:
            report += "✗ System performance needs improvement. Consider model refinement.\n"
        
        if len(failed_cases) <= 3:
            report += "✓ Number of failed cases is within acceptable range.\n"
        else:
            report += "⚠ Consider analyzing failed cases for systematic errors.\n"
        
        # Save report
        output_path = os.path.join(self.output_dir, output_filename)
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"Summary report saved: {output_path}")
        return report