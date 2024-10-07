import torch
import logging
import cv2
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

class SSD300Model:
    def __init__(self, model_path=None, device='cpu', confidence_threshold=0.25, iou_threshold=0.45):
        """
        Initializes the SSD300 model for detection (VGG16 backbone).

        Args:
            model_path (str): Path to the model weights (optional). If not provided, uses the default pre-trained weights.
            device (str): Device to run the model on ('cpu' or 'cuda').
            confidence_threshold (float): Minimum confidence for detections.
            iou_threshold (float): IoU threshold for matching.
        """
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.class_names = ['person']  # Assuming 'person' is the only class of interest

        self.model = self.load_model(model_path)
        self.reset_metrics()

        # Define image transformations (without resizing)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def set_confidence_threshold(self, threshold):
        """
        Sets the confidence threshold for detections.

        Args:
            threshold (float): Confidence threshold value.
        """
        self.confidence_threshold = threshold
        logging.info(f"Confidence threshold set to {self.confidence_threshold}.")

    def calculate_iou(self, boxA, boxB):
        """
        Calculates the Intersection over Union (IoU) between two bounding boxes.

        Args:
            boxA (list or np.ndarray): [x1, y1, x2, y2]
            boxB (list or np.ndarray): [x1, y1, x2, y2]

        Returns:
            float: IoU value
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interWidth = max(0, xB - xA)
        interHeight = max(0, yB - yA)
        interArea = interWidth * interHeight

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)

        return iou

    def load_model(self, model_path):
        """
        Loads the SSD300 model from the specified path.

        Args:
            model_path (str): Path to the SSD300 model weights.

        Returns:
            torch.nn.Module: Loaded SSD300 model.
        """
        try:
            weights = SSD300_VGG16_Weights.DEFAULT if model_path is None else None
            model = ssd300_vgg16(weights=weights)
            if model_path:
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                logging.info(f"SSD300 model loaded from {model_path}.")
            else:
                logging.info("SSD300 model loaded with pre-trained weights.")
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            logging.error(f"Failed to load SSD300 model: {e}", exc_info=True)
            raise

    def detect(self, frame):
        """
        Performs object detection on a single frame.

        Args:
            frame (numpy.ndarray): The input image in BGR format.

        Returns:
            dict: Detected boxes, scores, and labels.
        """
        try:
            # Convert BGR frame to RGB as the model expects RGB input
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Apply the transforms without resizing
            image = self.transform(image).to(self.device)

            # Pass the image through the model
            with torch.no_grad():
                outputs = self.model([image])  # Expect a list of outputs

            # Ensure we received an output
            if not outputs or len(outputs) == 0:
                logging.info("No detections made.")
                return None

            # Process the output
            outputs = outputs[0]  # Take the first (and only) result in the batch
            boxes = outputs['boxes'].detach().cpu().numpy()
            scores = outputs['scores'].detach().cpu().numpy()
            labels = outputs['labels'].detach().cpu().numpy().astype(int)

            logging.info(f"Detected labels: {labels}")
            logging.info(f"Detected scores: {scores}")

            # Map 'person' to 0 for consistency
            person_label_id = 1
            labels_mapped = np.where(labels == person_label_id, 0, -1)  # Map 'person' to 0, others to -1
            valid_indices = labels_mapped != -1
            boxes = boxes[valid_indices]
            scores = scores[valid_indices]
            labels = labels_mapped[valid_indices]

            # Filter detections based on the confidence threshold
            mask = scores >= self.confidence_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]

            if len(boxes) == 0:
                logging.info(f"No detections after filtering for confidence >= {self.confidence_threshold}.")
                return None

            # No need to rescale bounding boxes back to the original size since we did not resize the image
            return {
                'boxes': boxes,
                'scores': scores,
                'labels': labels
            }

        except Exception as e:
            logging.error(f"Detection failed: {e}", exc_info=True)
            return None

    def reset_metrics(self):
        """
        Resets the metrics for a new inference session.
        """
        self.metrics = {
            'TP': 0,
            'FP': 0,
            'FN': 0,
            'Average Confidence': 0.0,
            'IoU_Sum': 0.0,
            'Detections': []  # List to store all detections for mAP calculation
        }

    def update_metrics(self, detections, ground_truths):
        """
        Updates the metrics based on detections and ground truths.

        Args:
            detections (dict): Detections with boxes, scores, and labels.
            ground_truths (dict): Ground truths with boxes and labels.
        """
        if detections is None:
            logging.info("No detections to update metrics.")
            self.metrics['FN'] += len(ground_truths['boxes'])
            return

        detected_boxes = detections['boxes']
        detected_scores = detections['scores']
        detected_labels = detections['labels']

        gt_boxes = ground_truths['boxes']
        gt_labels = ground_truths['labels']

        # Map ground truth labels to match SSD labels
        # Assumption: 'person' label in MOT17 is 1 and in SSD is 0
        gt_labels_mapped = np.array([0 if label == 1 else -1 for label in gt_labels])

        logging.info(f"Ground truth labels (original): {gt_labels}")
        logging.info(f"Ground truth labels (mapped): {gt_labels_mapped}")

        # Filter out any ground truths that are not 'person' after mapping
        valid_gt_indices = gt_labels_mapped != -1
        gt_boxes = gt_boxes[valid_gt_indices]
        gt_labels = gt_labels_mapped[valid_gt_indices]

        matched_gt = set()

        for det_box, det_label, det_score in zip(detected_boxes, detected_labels, detected_scores):
            best_iou = 0
            best_gt_idx = -1

            for idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if idx in matched_gt:
                    continue
                if det_label != gt_label:
                    continue

                # Calculate IoU between predicted and ground truth box
                iou = self.calculate_iou(det_box, gt_box)
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_gt_idx = idx

            # If a good match is found, it's a True Positive (TP)
            if best_gt_idx >= 0:
                self.metrics['TP'] += 1
                self.metrics['IoU_Sum'] += best_iou
                matched_gt.add(best_gt_idx)
                self.metrics['Detections'].append({
                    'score': det_score,
                    'matched': True,
                    'iou': best_iou
                })
                logging.info(f"Match found: Detection label - {det_label}, IoU - {best_iou}")
            else:
                # If no match is found, it's a False Positive (FP)
                self.metrics['FP'] += 1
                self.metrics['Detections'].append({
                    'score': det_score,
                    'matched': False,
                    'iou': 0.0
                })
                logging.info(f"No match for detection: Label - {det_label}, Score - {det_score}")

        # Any ground truths not matched are False Negatives (FN)
        self.metrics['FN'] += len(gt_boxes) - len(matched_gt)
        logging.info(f"Total TP: {self.metrics['TP']}, FP: {self.metrics['FP']}, FN: {self.metrics['FN']}")

        # Update average confidence
        self.metrics['Average Confidence'] += np.sum(detected_scores)

    def compute_metrics(self):
        """
        Computes the final metrics after inference, including mAP.

        Returns:
            dict: Computed metrics.
        """
        # Sort detections by confidence score in descending order
        detections = sorted(self.metrics['Detections'], key=lambda x: x['score'], reverse=True)

        tp = 0
        fp = 0
        recall = []
        precision = []
        iou_list = []

        for det in detections:
            if det['matched']:
                tp += 1
                iou_list.append(det['iou'])
            else:
                fp += 1
            current_recall = tp / (tp + self.metrics['FN']) if (tp + self.metrics['FN']) > 0 else 0
            current_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall.append(current_recall)
            precision.append(current_precision)

        # Compute Average Precision (AP) using the 11-point interpolation method
        ap = self.calculate_average_precision(recall, precision)

        # Compute mean Average Precision (mAP) - since only one class, mAP = AP
        mAP = ap

        # Compute average IoU
        iou = self.metrics['IoU_Sum'] / self.metrics['TP'] if self.metrics['TP'] > 0 else 0

        # Compute average confidence
        avg_conf = self.metrics['Average Confidence'] / len(detections) if len(detections) > 0 else 0

        # Compute Precision, Recall, F1 Score
        precision_final = self.metrics['TP'] / (self.metrics['TP'] + self.metrics['FP']) if (self.metrics['TP'] + self.metrics['FP']) > 0 else 0
        recall_final = self.metrics['TP'] / (self.metrics['TP'] + self.metrics['FN']) if (self.metrics['TP'] + self.metrics['FN']) > 0 else 0
        f1_score = (2 * precision_final * recall_final) / (precision_final + recall_final) if (
            precision_final + recall_final) > 0 else 0

        self.metrics.update({
            'Precision': precision_final,
            'Recall': recall_final,
            'F1 Score': f1_score,
            'IoU': iou,
            'mAP': mAP,
            'Average Confidence': avg_conf
        })

        logging.info(f"Computed Metrics: {self.metrics}")

        return self.metrics

    def calculate_average_precision(self, recall, precision):
        """
        Calculates Average Precision (AP) using the 11-point interpolation method.

        Args:
            recall (list): List of recall values.
            precision (list): List of precision values.

        Returns:
            float: Average Precision.
        """
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            precisions = [p for r, p in zip(recall, precision) if r >= t]
            p = max(precisions) if precisions else 0
            ap += p / 11.0
        return ap

    def save_metrics_table(self, metrics, image_path):
        """
        Saves the metrics as a table image.

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
                ['Processing Time (s)', f"{metrics.get('Processing Time', 0.0):.2f} s"]
            ]

            # Create a figure and axis
            fig, ax = plt.subplots(figsize=(8, 4))  # Adjust size as needed
            ax.axis('off')  # Hide axes

            # Create the table
            table = plt.table(cellText=table_data,
                              colLabels=None,
                              cellLoc='center',
                              loc='center')

            # Format the table
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.2)

            # Highlight header row
            for key, cell in table.get_celld().items():
                if key[0] == 0:
                    cell.set_facecolor('#40466e')
                    cell.set_text_props(color='w')

            plt.tight_layout()
            plt.savefig(image_path, bbox_inches='tight')
            plt.close()
            logging.info(f"Metrics table saved to {image_path}.")
        except Exception as e:
            logging.error(f"Failed to save metrics table: {e}", exc_info=True)

    def save_confusion_matrix_image(self, metrics, image_path):
        """
        Saves a confusion matrix image showing TP, FP, FN counts.

        Args:
            metrics (dict): Dictionary containing 'TP', 'FP', 'FN'.
            image_path (str): Path to save the image.
        """
        try:
            categories = ['True Positives', 'False Positives', 'False Negatives']
            counts = [metrics.get('TP', 0), metrics.get('FP', 0), metrics.get('FN', 0)]
            colors = ['#4CAF50', '#F44336', '#2196F3']  # Green, Red, Blue

            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(categories, counts, color=colors)
            ax.set_title('Confusion Matrix')
            ax.set_ylabel('Count')

            # Annotate bars with counts
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.5, int(yval),
                        ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(image_path)
            plt.close()
            logging.info(f"Confusion matrix image saved to {image_path}.")
        except Exception as e:
            logging.error(f"Failed to save confusion matrix image: {e}", exc_info=True)

    def visualize_detections(self, frame, detections, save_path=None):
        """
        Visualizes detections by drawing bounding boxes and labels on the frame.

        Args:
            frame (numpy.ndarray): The input image in BGR format.
            detections (dict): Detections with boxes, scores, and labels.
            save_path (str): Path to save the visualized image (optional).

        Returns:
            numpy.ndarray: The image with visualized detections.
        """
        for box, score, label in zip(detections['boxes'], detections['scores'], detections['labels']):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label_text = f"{self.class_names[label]} {score:.2f}"
            cv2.putText(frame, label_text, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if save_path:
            cv2.imwrite(save_path, frame)
        return frame
