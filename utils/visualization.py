# utils/visualization.py

import logging
import cv2
import matplotlib.pyplot as plt
from matplotlib.table import Table
import numpy as np

def visualize_detections(frame, detections, class_names, algo, confidence_threshold=0.5):
    """
    Draws bounding boxes and labels on the frame based on detections.

    Args:
        frame (numpy.ndarray): The input image in BGR format.
        detections (dict): Detected boxes, scores, and labels.
        class_names (list): List of class names.
        algo (str): Algorithm name (e.g., 'YOLOv8s', 'Faster R-CNN', 'SSD').
        confidence_threshold (float): Minimum confidence for detections.

    Returns:
        numpy.ndarray: Annotated frame.
    """
    logging.debug(f"Drawing bounding boxes on the frame with {algo}")
    person_detection_count = 0

    if detections is not None:
        try:
            for box, score, label in zip(detections['boxes'], detections['scores'], detections['labels']):
                label_index = int(label)
                # Since we mapped 'person' label to 0 for all models
                if label_index != 0:
                    continue

                if score < confidence_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box)
                conf = score
                label_name = class_names[label_index]
                label_text = f"{label_name} {conf * 100:.2f}%"

                box_color = (0, 255, 0)  # Use green color for all

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, box_color, 2)
                person_detection_count += 1

            # Display the number of detected persons on the frame
            text = f"Persons detected: {person_detection_count}"
            font_scale = 1
            thickness = 2
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_width, text_height = text_size

            # Position the text in the top-right corner
            x = frame.shape[1] - text_width - 10
            y = text_height + 10

            # Draw a rectangle background for better visibility
            cv2.rectangle(frame, (x - 5, y - text_height - 5), (x + text_width + 5, y + 5), (0, 0, 0), -1)

            # Put the text on the frame
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

            logging.debug(f"{algo} detected persons: {person_detection_count}")

            # Check if TP > 0 to avoid division by zero
            if person_detection_count > 0:
                logging.info(f"IoU Calculation: TP > 0, proceeding with calculations.")
            else:
                logging.warning(f"IoU Calculation: TP = 0, skipping average IoU computation.")

        except Exception as e:
            logging.error(f"Error in visualize_detections: {e}", exc_info=True)

    return frame

def save_metrics_table(metrics, image_path):
    """
    Saves the metrics as a horizontally oriented table image with proper formatting.

    Args:
        metrics (dict): Dictionary containing computed metrics.
        image_path (str): Path to save the image.
    """
    try:
        table_data = [
            ['Metric', 'Value'],
            ['mAP', f"{metrics.get('mAP', 0.0) * 100:.2f}%"],
            ['Precision', f"{metrics.get('Precision', 0.0) * 100:.2f}%"],
            ['Recall', f"{metrics.get('Recall', 0.0) * 100:.2f}%"],
            ['F1 Score', f"{metrics.get('F1 Score', 0.0) * 100:.2f}%"],
            ['IoU', f"{metrics.get('IoU', 0.0) * 100:.2f}%"],
            ['Average Confidence', f"{metrics.get('Average Confidence', 0.0):.2f}"],
            ['Processing Time (s)', f"{metrics.get('Processing Time (s)', 0.0):.2f} s"],
            ['TP', f"{metrics.get('TP', 0)}"],
            ['FP', f"{metrics.get('FP', 0)}"],
            ['FN', f"{metrics.get('FN', 0)}"]
        ]

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 5))  # Adjust size as needed
        ax.axis('off')  # Hide axes

        # Create the table
        table = plt.table(cellText=table_data,
                          colLabels=None,
                          cellLoc='center',
                          loc='center')

        # Adjust table properties
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)  # Adjust scaling as needed for better readability

        # Make header bold and set background color
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#f2f2f2')  # Light gray background for header
            else:
                cell.set_facecolor('#ffffff')  # White background for data

        plt.tight_layout()
        plt.savefig(image_path, bbox_inches='tight')
        plt.close()
        logging.debug(f"Metrics table saved to {image_path}.")
    except Exception as e:
        logging.error(f"Failed to save metrics table: {e}", exc_info=True)

def save_confusion_matrix_image(metrics, image_path):
    """
    Saves a confusion matrix image showing TP, FP, FN counts.

    Args:
        metrics (dict): Dictionary containing 'TP', 'FP', 'FN'.
        image_path (str): Path to save the image.
    """
    try:
        if 'TP' in metrics and 'FP' in metrics and 'FN' in metrics:
            categories = ['True Positives', 'False Positives', 'False Negatives']
            counts = [metrics.get('TP', 0), metrics.get('FP', 0), metrics.get('FN', 0)]
            colors = ['#4CAF50', '#F44336', '#2196F3']  # Green, Red, Blue

            fig, ax = plt.subplots(figsize=(8, 6))  # Increased height for better readability
            bars = ax.bar(categories, counts, color=colors)
            ax.set_title('Confusion Matrix', fontsize=16)
            ax.set_ylabel('Count', fontsize=14)
            ax.set_xlabel('Categories', fontsize=14)

            # Annotate bars with counts
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + max(counts)*0.01, int(yval),
                        ha='center', va='bottom', fontsize=12)

            # Improve layout
            plt.tight_layout()
            plt.savefig(image_path)
            plt.close()
            logging.debug(f"Confusion matrix image saved to {image_path}.")
        else:
            logging.warning("TP, FP, FN values are not available. Skipping confusion matrix generation.")
    except Exception as e:
        logging.error(f"Failed to save confusion matrix image: {e}", exc_info=True)
