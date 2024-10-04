import logging
import cv2
import torch
from torchvision import models, transforms
import numpy as np
from matplotlib.table import Table
import matplotlib.pyplot as plt

class FasterRCNNModel:
    def __init__(self, device='cpu', confidence_threshold=0.5, iou_threshold=0.5):

        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = self.load_model()

        # Get model class names (COCO dataset)
        self.class_names = ['__background__', 'person']  # Simplified for MOT17
        logging.info(f"Model class names: {self.class_names}")

        # Initialize metrics
        self.reset_metrics()

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def to(self, device):
        self.device = torch.device(device)
        self.model.to(self.device)
        logging.info(f"Model is set to {self.device}.")
        return self

    def set_confidence_threshold(self, threshold):
        self.confidence_threshold = threshold
        logging.info(f"Confidence threshold set to {self.confidence_threshold}.")

    def load_model(self):
        """
        Loads the Faster R-CNN model from torchvision.
        Returns:
            torch.nn.Module: Loaded Faster R-CNN model.
        """
        try:
            model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            model.eval()
            model.to(self.device)
            logging.info("Faster R-CNN model loaded successfully.")
            return model
        except Exception as e:
            logging.error(f"Failed to load Faster R-CNN model: {e}", exc_info=True)
            raise

    def detect_batch(self, frames):
        """
        Performs object detection on a batch of frames.
        Args:
            frames (list of numpy.ndarray): List of input images in BGR format.
        Returns:
            list: List of detection dictionaries for each frame in the batch.
        """
        try:
            batch_images = [self.transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).to(self.device) for frame in frames]
            batch_images = torch.stack(batch_images)

            with torch.no_grad():
                outputs = self.model(batch_images)

            detections_batch = []
            for output in outputs:
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy().astype(int)

                valid_indices = scores >= self.confidence_threshold
                boxes = boxes[valid_indices]
                scores = scores[valid_indices]
                labels = labels[valid_indices]

                detections_batch.append({
                    'boxes': boxes,
                    'scores': scores,
                    'labels': labels
                })

            return detections_batch
        except Exception as e:
            logging.error(f"Batch detection failed: {e}", exc_info=True)
            return None

    def detect(self, frame):
        """
        Performs object detection on a single frame by wrapping it in a batch of one.
        Args:
            frame (numpy.ndarray): The input image in BGR format.
        Returns:
            dict: Detected boxes, scores, and labels.
        """
        detections_batch = self.detect_batch([frame])
        if detections_batch is not None and len(detections_batch) > 0:
            return detections_batch[0]
        return None

    def calibrate_confidence_threshold(self, frames, ground_truths):
        """
        Calibrates the confidence threshold using a batch of 10 frames and tests thresholds
        from 0.3 to 0.75 in steps of 0.1.

        Args:
            frames (list of numpy.ndarray): List of frames for calibration.
            ground_truths (list of dict): List of ground truth data corresponding to the frames.

        Returns:
            float: Optimal confidence threshold.
        """
        try:
            calibration_frames = frames[:10]  # Use the first 10 frames for calibration
            calibration_ground_truths = ground_truths[:10]

            best_threshold = 0.5
            best_mAP = 0.0
            thresholds_to_test = np.arange(0.4, 0.80, 0.1)  # 0.3 to 0.75 with step 0.1

            for threshold in thresholds_to_test:
                print(f"Testing threshold: {threshold:.2f}")  # Debugging: Print the threshold being tested
                self.set_confidence_threshold(threshold)

                # Detect objects for the calibration frames using the current threshold
                detections = self.detect_batch(calibration_frames)

                # If no valid detections, skip this threshold
                if detections is None or len(detections) == 0:
                    logging.warning(f"No valid detections for threshold {threshold}.")
                    continue

                # Compute mAP for the detections vs. ground truth
                mAP = self.compute_map(detections, calibration_ground_truths)

                # Update the best threshold if a higher mAP is found
                if mAP > best_mAP:
                    best_mAP = mAP
                    best_threshold = threshold

            logging.info(f"Optimal confidence threshold found: {best_threshold:.2f} with mAP: {best_mAP:.4f}")
            return best_threshold

        except Exception as e:
            logging.error(f"Calibration failed: {e}", exc_info=True)
            return 0.5

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

        gt_labels_mapped = np.array([1 if label == 1 else -1 for label in gt_labels])

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

                iou = self.calculate_iou(det_box, gt_box)
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_gt_idx = idx

            if best_gt_idx >= 0:
                self.metrics['TP'] += 1
                self.metrics['IoU_Sum'] += best_iou
                matched_gt.add(best_gt_idx)
                self.metrics['Detections'].append({
                    'score': det_score,
                    'matched': True,
                    'iou': best_iou
                })
            else:
                self.metrics['FP'] += 1
                self.metrics['Detections'].append({
                    'score': det_score,
                    'matched': False,
                    'iou': 0.0
                })

        self.metrics['FN'] += len(gt_boxes) - len(matched_gt)
        self.metrics['Average Confidence'] += np.sum(detected_scores)

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
            'Detections': []
        }

    def compute_metrics(self):
        """
        Computes the final metrics after inference, including mAP.
        Returns:
            dict: Computed metrics.
        """
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

        ap = self.calculate_average_precision(recall, precision)
        mAP = ap
        iou = self.metrics['IoU_Sum'] / self.metrics['TP'] if self.metrics['TP'] > 0 else 0
        avg_conf = self.metrics['Average Confidence'] / len(detections) if len(detections) > 0 else 0

        precision_final = self.metrics['TP'] / (self.metrics['TP'] + self.metrics['FP']) if (self.metrics['TP'] +
                                                                                             self.metrics['FP']) > 0 else 0
        recall_final = self.metrics['TP'] / (self.metrics['TP'] + self.metrics['FN']) if (self.metrics['TP'] +
                                                                                          self.metrics['FN']) > 0 else 0
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

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.axis('off')

            table = Table(ax, bbox=[0, 0, 1, 1])
            nrows, ncols = len(table_data), len(table_data[0])
            width, height = 1.0 / ncols, 1.0 / nrows

            for i in range(nrows):
                for j in range(ncols):
                    cell = table_data[i][j]
                    table.add_cell(i, j, width, height, text=cell, loc='center',
                                   facecolor='lightgrey' if i == 0 else 'white')

            ax.add_table(table)
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
            colors = ['#4CAF50', '#F44336', '#2196F3']

            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(categories, counts, color=colors)
            ax.set_title('Confusion Matrix')
            ax.set_ylabel('Count')

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
