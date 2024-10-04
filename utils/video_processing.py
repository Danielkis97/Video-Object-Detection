import cv2
import torch
import os
import json
import logging
from tqdm import tqdm
from collections import defaultdict
from utils.visualization import visualize_detections, save_metrics_table, save_confusion_matrix_image
import pandas as pd

def annotate_frames(frames_dir, annotated_dir, model, algo, confidence_threshold=0.5, object_count=None):
    """
    Annotates frames with detected objects using the provided model.

    Args:
        frames_dir (str): Directory containing original frames.
        annotated_dir (str): Directory to save annotated frames.
        model (object): Initialized object detection model.
        algo (str): Algorithm name ('YOLOv8s', 'Faster R-CNN', 'SSD').
        confidence_threshold (float): Minimum confidence to consider a detection.
        object_count (defaultdict): Dictionary to count detected objects.

    Returns:
        None
    """
    os.makedirs(annotated_dir, exist_ok=True)
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    logging.info(f"Annotating {len(frame_files)} frames in {frames_dir}")

    for frame_file in tqdm(frame_files, desc="Annotating Frames"):
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)

        if frame is None:
            logging.warning(f"Failed to read {frame_path}. Skipping frame.")
            continue

        # Run object detection
        detections = model.detect(frame)

        if detections is not None:
            # Set the confidence threshold in visualization
            annotated_frame = visualize_detections(frame, detections, model.class_names, algo, confidence_threshold)
        else:
            annotated_frame = frame

        # Save annotated frame
        annotated_frame_path = os.path.join(annotated_dir, frame_file)
        cv2.imwrite(annotated_frame_path, annotated_frame)
        logging.info(f"Annotated frame saved to {annotated_frame_path}")

        # Count detected objects (only 'person' class)
        if detections and object_count is not None:
            # Assuming class_id for 'person' is 0 for YOLOv8 and 1 for other models
            if algo == "YOLOv8s":
                person_class_id = 0  # Verify the actual class ID in YOLOv8
            else:
                person_class_id = 1  # Typically, class 1 is 'person' for Faster R-CNN and SSD

            person_detections = sum(label == person_class_id for label in detections['labels'])
            object_count['person'] += person_detections

def process_video(input_path, model, algo, progress_bar=None, output_dir='', annotations_file='', stop_event=None):
    """
    Processes a video with a given model and algorithm, annotates detected objects, and saves the results.

    Args:
        input_path (str): Path to the input video.
        model (object): The object detection model.
        algo (str): Name of the algorithm (e.g., 'YOLOv8s', 'Faster R-CNN', 'SSD').
        progress_bar (ttk.Progressbar, optional): GUI progress bar.
        output_dir (str): Directory to save outputs.
        annotations_file (str): Path to the Ground-Truth annotations in MOT17 format.
        stop_event (threading.Event, optional): Event to signal stopping the process.

    Returns:
        dict: Dictionary containing the path to the annotated video and computed metrics.
    """
    logging.info(f"Processing video: {input_path} with model: {algo}")

    # Determine the device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.device = device
    logging.info(f"Using device: {device}")

    # Open the video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video file {input_path}")
        return {}

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logging.info(f"Video properties: {frame_count} frames, {fps} FPS, {width}x{height} pixels")
    print(f"Video properties: {frame_count} frames, {fps} FPS, {width}x{height} pixels")  # Debugging

    # Get video name without extension
    video_name = os.path.splitext(os.path.basename(input_path))[0]

    # Create output video name and path
    output_video_name = f"{video_name}annotated{algo}.mp4"
    annotated_videos_dir = os.path.join(output_dir, 'annotated_videos')
    os.makedirs(annotated_videos_dir, exist_ok=True)
    output_video_path = os.path.join(annotated_videos_dir, output_video_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if not out.isOpened():
        logging.error(f"VideoWriter could not be opened with resolution: {(width, height)}")
        cap.release()
        return {}

    frame_idx = 0
    confidence_threshold = model.confidence_threshold  # Adjustable confidence threshold

    # Load Ground Truth Annotations from MOT17
    ground_truths = load_ground_truths(annotations_file)
    logging.info(f"Loaded ground truths for {len(ground_truths)} frames.")
    print(f"Loaded ground truths for {len(ground_truths)} frames.")  # Debugging

    # Initialize object count
    object_count = defaultdict(int)

    while True:
        if stop_event and stop_event.is_set():
            logging.info("Stop event detected. Terminating video processing.")
            break

        ret, frame = cap.read()
        if not ret:
            break

        logging.info(f"Processing frame {frame_idx + 1}/{frame_count} ({algo})")
        print(f"Processing frame {frame_idx + 1}/{frame_count} ({algo})")  # Debugging

        # Perform detection
        detections = model.detect(frame)

        # Get corresponding ground truth for this frame
        gt = ground_truths.get(frame_idx, None)

        # Update metrics
        model.update_metrics(detections, gt)

        # Annotate frame
        if algo == "YOLOv8s":
            # For YOLOv8s, 'detections' is a dict. Modify it to match expected attributes.
            class Detection:
                def _init_(self, boxes, conf, cls):
                    self.boxes = type('Boxes', (object,), {'xyxy': boxes, 'conf': conf, 'cls': cls})

            detection_obj = Detection(detections['boxes'], detections['scores'], detections['labels'])
            annotated_frame = visualize_detections(frame, detection_obj, model.class_names, algo,
                                                   model.confidence_threshold)
        else:
            annotated_frame = visualize_detections(frame, detections, model.class_names, algo,
                                                   model.confidence_threshold)

        # Write the frame to the output video
        out.write(annotated_frame)
        logging.info(f"Frame {frame_idx + 1} written to output video")
        print(f"Frame {frame_idx + 1} written to output video")  # Debugging

        # Update progress bar
        if progress_bar is not None:
            progress = ((frame_idx + 1) / frame_count) * 100
            progress_bar['value'] = progress
            progress_bar.update()

        # Additional Logging for SSD and Faster R-CNN
        if algo in ["SSD", "Faster R-CNN"]:
            logging.info(f"[{algo}] Currently processing frame {frame_idx + 1}.")
            print(f"[{algo}] Currently processing frame {frame_idx + 1}.")  # Debugging

        frame_idx += 1

    # Finalize progress bar
    if progress_bar is not None:
        progress_bar['value'] = 100
        progress_bar.update()
        logging.info("Progress: 100.00%")
        print("Progress: 100.00%")  # Debugging

    # Release resources
    cap.release()
    out.release()

    # Compute final metrics
    metrics = model.compute_metrics()
    logging.info(f"Final Metrics for {algo}: {metrics}")
    print(f"Final Metrics for {algo}: {metrics}")  # Debugging

    # Add processing time to metrics
    metrics['Processing Time (s)'] = round(model.metrics.get('Processing Time (s)', 0.0), 2)

    # Add best and worst confidence thresholds
    # Ensure that best_threshold and worst_threshold are correctly set
    if hasattr(model, 'best_threshold') and hasattr(model, 'worst_threshold'):
        metrics['Best Confidence Threshold'] = round(model.best_threshold, 2)
        metrics['Worst Confidence Threshold'] = round(model.worst_threshold, 2)
    else:
        # If calibration was not performed, use the current confidence threshold
        metrics['Best Confidence Threshold'] = round(confidence_threshold, 2)
        metrics['Worst Confidence Threshold'] = round(confidence_threshold, 2)

    # Compute average confidence in 0.xx format
    metrics['Average Confidence'] = round(metrics.get('Average Confidence', 0.0), 2)

    # Optionally, save metrics to a JSON file
    metrics_path = os.path.join(annotated_videos_dir, f"{video_name}_{algo}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Metrics saved to {metrics_path}")
    print(f"Metrics saved to {metrics_path}")  # Debugging

    # Save metrics as a table
    metrics_image_path = os.path.join(annotated_videos_dir, f"{video_name}_{algo}_metrics_table.png")
    save_metrics_table(metrics, metrics_image_path)
    logging.info(f"Metrics table saved to {metrics_image_path}.")
    print(f"Metrics table saved to {metrics_image_path}.")  # Debugging

    # Save the confusion matrix as an image
    cm_image_path = os.path.join(annotated_videos_dir, f"{video_name}_{algo}_confusion_matrix.png")
    save_confusion_matrix_image(metrics, cm_image_path)
    logging.info(f"Confusion matrix image saved to {cm_image_path}.")
    print(f"Confusion matrix image saved to {cm_image_path}.")  # Debugging

    # Optionally, return the paths
    return {
        'annotated_video_path': output_video_path,
        'metrics_json_path': metrics_path,
        'metrics_image_path': metrics_image_path,
        'confusion_matrix_image_path': cm_image_path,
        'metrics': metrics
    }

def load_ground_truths(gt_file):
    """
    Loads ground truth annotations from the MOT17 dataset.

    Args:
        gt_file (str): Path to the 'gt/gt.txt' file.

    Returns:
        dict: Dictionary mapping frame indices to their ground truth annotations.
    """
    if not os.path.isfile(gt_file):
        logging.error(f"Ground truth file not found at {gt_file}")
        return {}

    try:
        # Read the ground truth file
        gt_data = pd.read_csv(gt_file, header=None)
        num_columns = gt_data.shape[1]

        # Adjust the expected number of columns if necessary
        if num_columns == 10:
            gt_data.columns = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width',
                               'bb_height', 'conf', 'class', 'visibility', 'x']
        elif num_columns == 9:
            gt_data.columns = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width',
                               'bb_height', 'conf', 'class', 'visibility']
        else:
            logging.warning(f"Unexpected number of columns in gt.txt: {num_columns}")
            # Attempt to assign default column names
            gt_data = gt_data.iloc[:, :9]
            gt_data.columns = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width',
                               'bb_height', 'conf', 'class', 'visibility']

        # Drop rows with missing values in essential columns
        essential_columns = ['frame', 'bb_left', 'bb_top', 'bb_width', 'bb_height']
        gt_data.dropna(subset=essential_columns, inplace=True)

        # Convert data types
        gt_data[essential_columns] = gt_data[essential_columns].astype(float)

        # Filter out non-Ground-Truth entries if necessary
        if 'conf' in gt_data.columns:
            gt_data = gt_data[gt_data['conf'] == 1]

        # Assign class labels (1 for 'person')
        gt_data['class'] = 1

        ground_truths = {}
        for frame_idx in gt_data['frame'].unique():
            frame_data = gt_data[gt_data['frame'] == frame_idx]
            boxes = frame_data[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values
            # Convert [x, y, w, h] to [x1, y1, x2, y2]
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = x1 + w
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2 = y1 + h
            labels = frame_data['class'].tolist()
            ground_truths[int(frame_idx) - 1] = {'boxes': boxes.tolist(), 'labels': labels}  # Adjust frame index to start from 0

        logging.info(f"Loaded ground truth for {len(ground_truths)} frames.")
        print(f"Loaded ground truths for {len(ground_truths)} frames.")  # Debugging
        return ground_truths

def visualize_detections(frame, detections, class_names, algo, confidence_threshold=0.5):
    """
    Draws bounding boxes and labels on the frame based on detections.

    Args:
        frame (numpy.ndarray): The input image in BGR format.
        detections (dict or object): Detected boxes, scores, and labels.
        class_names (list): List of class names.
        algo (str): Algorithm name (e.g., 'YOLOv8s', 'Faster R-CNN', 'SSD').
        confidence_threshold (float): Minimum confidence for detections.

    Returns:
        numpy.ndarray: Annotated frame.
    """
    logging.info(f"Drawing bounding boxes on the frame with {algo}")
    person_detection_count = 0

    if detections is not None:
        try:
            if algo == "YOLOv8s":
                # Expecting detections to have 'boxes.xyxy', 'boxes.conf', 'boxes.cls'
                for box, score, label in zip(detections.boxes.xyxy, detections.boxes.conf, detections.boxes.cls):
                    if algo in ["Faster R-CNN", "SSD"]:
                        # For models where labels start at 1
                        label = int(label) - 1  # Adjust label index
                    else:
                        label = int(label)  # For YOLOv8s, labels start at 0

                    if label < 0 or label >= len(class_names):
                        continue  # Invalid label index

                    if score < confidence_threshold:
                        continue

                    xmin, ymin, xmax, ymax = map(int, box.cpu().numpy())
                    conf = score.cpu().numpy()
                    label_name = class_names[label]

                    # Only process 'person' class
                    if label_name != 'person':
                        continue

                    label_text = f"{label_name} {conf * 100:.2f}%"
                    box_color = (0, 255, 0)  # Green in BGR

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, 2)
                    cv2.putText(frame, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, box_color, 2)
                    person_detection_count += 1

            else:  # For Faster R-CNN and SSD
                if 'boxes' not in detections or 'scores' not in detections or 'labels' not in detections:
                    logging.error(f"{algo} detections do not have the expected keys 'boxes', 'scores', and 'labels'")
                    return frame

                for box, score, label in zip(detections['boxes'], detections['scores'], detections['labels']):
                    # Adjust label index if necessary
                    label = int(label) - 1  # Adjust label index for models where labels start at 1

                    if label < 0 or label >= len(class_names):
                        continue  # Invalid label index

                    if score < confidence_threshold:
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    conf = score
                    label_name = class_names[label]

                    # Only process 'person' class
                    if label_name != 'person':
                        continue

                    label_text = f"{label_name} {conf * 100:.2f}%"
                    box_color = (255, 0, 0)  # Blue in BGR

                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, box_color, 2)
                    person_detection_count += 1

            # Display the number of detected persons on the frame
            try:
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

                logging.info(f"{algo} detected persons: {person_detection_count}")
            except Exception as e:
                logging.error(f"Error while adding detection count to frame: {e}")

        except Exception as e:
            logging.error(f"Error in visualize_detections: {e}")

    return frame

def save_metrics_table(metrics, image_path):
    """
    Saves the metrics as a horiizontally oriented table image with proper formatting.

    Args:
        metrics (dict): Dictionary containing computed metrics.
        image_path (str): Path to save the image.
    """
    try:
        # Prepare data
        labels = ['mAP', 'Precision', 'Recall', 'F1 Score', 'IoU', 'Processing Time (s)', 'Average Confidence', 'Best Confidence Threshold', 'Worst Confidence Threshold']
        values = [
            f"{metrics.get('mAP', 0) * 100:.2f}%",
            f"{metrics.get('Precision', 0) * 100:.2f}%",
            f"{metrics.get('Recall', 0) * 100:.2f}%",
            f"{metrics.get('F1 Score', 0) * 100:.2f}%",
            f"{metrics.get('IoU', 0) * 100:.2f}%",
            f"{metrics.get('Processing Time (s)', 0):.2f} s",
            f"{metrics.get('Average Confidence', 0):.2f}",
            f"{metrics.get('Best Confidence Threshold', 0):.2f}",
            f"{metrics.get('Worst Confidence Threshold', 0):.2f}"
        ]

        # Draw table
        fig, ax = plt.subplots(figsize=(16, 3))  # Landscape orientation
        ax.axis('off')
        table = ax.table(cellText=[values],
                         colLabels=labels,
                         cellLoc='center',
                         loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)

        plt.tight_layout()
        plt.savefig(image_path)
        plt.close()
        logging.info(f"Metrics table saved to {image_path}.")
    except Exception as e:
        logging.error(f"Failed to save metrics table: {e}")

def save_confusion_matrix_image(metrics, image_path):
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

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(categories, counts, color=colors)
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('Count')

        # Annotate bars with counts
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, int(yval),
                    ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(image_path)
        plt.close()
        logging.info(f"Confusion matrix image saved to {image_path}.")
    except Exception as e:
        logging.error(f"Failed to save confusion matrix image: {e}")