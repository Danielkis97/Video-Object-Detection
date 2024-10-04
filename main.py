# main.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import logging
import os
import glob
import configparser
import time
import multiprocessing
import torch
import pandas as pd
import numpy as np
import cv2  # OpenCV imported
import json
import warnings
import sys

from utils.visualization import visualize_detections, save_metrics_table, save_confusion_matrix_image
from custom_models.yolo import YOLOv8Model
from custom_models.ssd import SSD300Model
from custom_models.custom_faster_rcnn import FasterRCNNModel
from torchvision.ops import box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated")
warnings.filterwarnings("ignore", message="Arguments other than a weight enum or `None` for 'weights' are deprecated")



logging.basicConfig(
    filename='app.log',
    filemode='w',
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def process_frame_for_calibration(args):
    model, frame, ground_truth = args
    device = model.device

    try:
        detections = model.detect(frame)
        if detections is None:
            detections = {'boxes': [], 'scores': [], 'labels': []}

        if len(detections['boxes']) > 0:
            preds = [{
                'boxes': torch.tensor(detections['boxes'], dtype=torch.float32).to(model.device),
                'scores': torch.tensor(detections['scores'], dtype=torch.float32).to(model.device),
                'labels': torch.tensor(detections['labels'], dtype=torch.int64).to(model.device)
            }]
        else:
            preds = [{
                'boxes': torch.empty((0, 4), dtype=torch.float32).to(model.device),
                'scores': torch.tensor([], dtype=torch.float32).to(model.device),
                'labels': torch.tensor([], dtype=torch.int64).to(model.device)
            }]

        if ground_truth and len(ground_truth['boxes']) > 0:
            targets = [{
                'boxes': torch.tensor(ground_truth['boxes'], dtype=torch.float32).to(model.device),
                'labels': torch.tensor(ground_truth['labels'], dtype=torch.int64).to(model.device)
            }]
        else:
            targets = [{
                'boxes': torch.empty((0, 4), dtype=torch.float32).to(model.device),
                'labels': torch.tensor([], dtype=torch.int64).to(model.device)
            }]

        if not preds[0]['boxes'].shape[0]:
            logging.warning("No predictions in calibration frame.")
        if not targets[0]['boxes'].shape[0]:
            logging.warning("No ground truth boxes in calibration frame.")

        return preds, targets
    except Exception as e:
        logging.error(f"Error in process_frame_for_calibration: {e}", exc_info=True)
        return [], []


class ObjectRecognitionApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Object Recognition Application")
        self.geometry("1600x1000")

        self.sequence_paths = []
        self.model_vars = {}
        self.stop_event = threading.Event()
        self.queue = queue.Queue()

        self.processing_time = {}
        self.metrics_storage = {}

        self.create_widgets()
        self.after(100, self.process_queue)

    def create_widgets(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Sequence Selection
        sequence_frame = ttk.LabelFrame(main_frame, text="Sequence Selection")
        sequence_frame.pack(fill=tk.X, pady=5)

        self.sequence_listbox = tk.Listbox(sequence_frame, selectmode=tk.MULTIPLE, height=5)
        self.sequence_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        sequence_scrollbar = ttk.Scrollbar(sequence_frame, orient="vertical", command=self.sequence_listbox.yview)
        self.sequence_listbox.configure(yscrollcommand=sequence_scrollbar.set)
        sequence_scrollbar.pack(side=tk.LEFT, fill=tk.Y)

        sequence_buttons_frame = ttk.Frame(sequence_frame)
        sequence_buttons_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        ttk.Button(sequence_buttons_frame, text="Add Sequences", command=self.browse_sequences).pack(fill=tk.X, pady=2)
        ttk.Button(sequence_buttons_frame, text="Remove Selected", command=self.remove_selected_sequences).pack(fill=tk.X, pady=2)
        ttk.Button(sequence_buttons_frame, text="Clear List", command=self.clear_sequences).pack(fill=tk.X, pady=2)

        # Model Selection
        model_frame = ttk.LabelFrame(main_frame, text="Model Selection")
        model_frame.pack(fill=tk.X, pady=5)

        self.model_options = ["YOLOv8s", "Faster R-CNN", "SSD"]
        self.model_vars = {model: tk.BooleanVar(value=False) for model in self.model_options}

        for model, var in self.model_vars.items():
            ttk.Checkbutton(model_frame, text=model, variable=var).pack(side=tk.LEFT, padx=5, pady=5)

        # Parameters
        params_frame = ttk.LabelFrame(main_frame, text="Parameters")
        params_frame.pack(fill=tk.X, pady=5)

        self.auto_threshold_var = tk.BooleanVar(value=True)
        self.auto_threshold_checkbox = ttk.Checkbutton(
            params_frame,
            text="Automatic Calibration",
            variable=self.auto_threshold_var,
            command=self.toggle_threshold_entry
        )
        self.auto_threshold_checkbox.pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Label(params_frame, text="Confidence Threshold:").pack(side=tk.LEFT, padx=5, pady=5)
        self.conf_threshold = tk.DoubleVar(value=0.5)
        self.conf_threshold_entry = ttk.Entry(params_frame, textvariable=self.conf_threshold, width=10)
        self.conf_threshold_entry.pack(side=tk.LEFT, padx=5, pady=5)

        calibration_method_frame = ttk.Frame(params_frame)
        calibration_method_frame.pack(side=tk.LEFT, padx=5, pady=5)

        self.calibration_method_var = tk.StringVar(value="linear")

        self.linear_search_radio = ttk.Radiobutton(
            calibration_method_frame,
            text="Linear Search",
            variable=self.calibration_method_var,
            value="linear"
        )
        self.linear_search_radio.pack(side=tk.LEFT, padx=2)

        self.binary_search_radio = ttk.Radiobutton(
            calibration_method_frame,
            text="Binary Search",
            variable=self.calibration_method_var,
            value="binary"
        )
        self.binary_search_radio.pack(side=tk.LEFT, padx=2)

        explanation_label = ttk.Label(
            params_frame,
            text="Choose calibration method: Binary Search (faster) or Linear Search (standard, more accurate).",
            wraplength=600
        )
        explanation_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.toggle_threshold_entry()

        # Control Buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)

        self.start_button = ttk.Button(control_frame, text="Start Processing", command=self.start_processing)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.stop_button = ttk.Button(control_frame, text="Stop Processing", command=self.stop_processing)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.stop_button.config(state='disabled')

        # Progress Bar
        self.progress = ttk.Progressbar(main_frame, orient='horizontal', mode='determinate')
        self.progress.pack(fill=tk.X, pady=5)

        # Status Display
        status_frame = ttk.LabelFrame(main_frame, text="Status")
        status_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.status_display = tk.Text(status_frame, height=15, state='disabled', wrap='word')
        self.status_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        status_scrollbar = ttk.Scrollbar(status_frame, orient="vertical", command=self.status_display.yview)
        self.status_display.configure(yscrollcommand=status_scrollbar.set)
        status_scrollbar.pack(side=tk.LEFT, fill=tk.Y)

        # Evaluation Results
        results_frame = ttk.LabelFrame(main_frame, text="Evaluation Results")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.results_notebook = ttk.Notebook(results_frame)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)

        self.results_tabs = {}

    def toggle_threshold_entry(self):
        if self.auto_threshold_var.get():
            self.conf_threshold_entry.config(state='disabled')
            self.linear_search_radio.config(state='normal')
            self.binary_search_radio.config(state='normal')
        else:
            self.conf_threshold_entry.config(state='normal')
            self.linear_search_radio.config(state='disabled')
            self.binary_search_radio.config(state='disabled')
            self.calibration_method_var.set("linear")

    def process_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                self.status_display.config(state='normal')
                self.status_display.insert(tk.END, msg + "\n")
                if self.is_scrollbar_at_bottom():
                    self.status_display.see(tk.END)
                self.status_display.config(state='disabled')
        except queue.Empty:
            pass
        self.after(100, self.process_queue)

    def is_scrollbar_at_bottom(self):
        return self.status_display.yview()[1] == 1.0

    def browse_sequences(self):
        selected_folder = filedialog.askdirectory(mustexist=True)
        if selected_folder:
            self.sequence_paths.append(selected_folder)
            self.sequence_listbox.insert(tk.END, selected_folder)
            logging.info(f"Added sequence: {selected_folder}")
            self.queue.put(f"Added sequence: {selected_folder}")

    def remove_selected_sequences(self):
        selected_indices = self.sequence_listbox.curselection()
        for index in reversed(selected_indices):
            self.sequence_listbox.delete(index)
            del self.sequence_paths[index]
        logging.info("Removed selected sequences.")
        self.queue.put("Removed selected sequences.")

    def clear_sequences(self):
        self.sequence_listbox.delete(0, tk.END)
        self.sequence_paths.clear()
        logging.info("Cleared all sequences.")
        self.queue.put("Cleared all sequences.")

    def start_processing(self):
        selected_models = [model for model, var in self.model_vars.items() if var.get()]
        if not self.sequence_paths or not selected_models:
            messagebox.showwarning("Input Required", "Please select at least one sequence and one model.")
            return

        self.stop_event.clear()
        self.disable_ui()
        self.status_display.config(state='normal')
        self.status_display.delete(1.0, tk.END)
        self.status_display.config(state='disabled')

        self.metrics_storage = {}

        threading.Thread(target=self.process_sequences, args=(self.sequence_paths, selected_models), daemon=True).start()

    def process_sequences(self, sequence_paths, selected_models):
        try:
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)

            total_frames_all = 0
            for sequence_folder in sequence_paths:
                img_folder = os.path.join(sequence_folder, 'img1')
                img_files = sorted(glob.glob(os.path.join(img_folder, '*.jpg')))
                total_frames_all += len(img_files)

            total_tasks = len(selected_models) * total_frames_all
            processed_tasks = 0

            LABEL_MAP = {
                "YOLOv8s": {"person": 0},
                "Faster R-CNN": {"person": 0},
                "SSD": {"person": 0}
            }

            for sequence_folder in sequence_paths:
                sequence_name = os.path.basename(sequence_folder)
                for algo in selected_models:
                    if self.stop_event.is_set():
                        self.queue.put("Processing stopped by user.")
                        logging.info("Processing stopped by user.")
                        break

                    self.queue.put(f"Processing sequence '{sequence_name}' with {algo}...")
                    logging.info(f"Processing sequence '{sequence_name}' with {algo}...")

                    try:
                        model = self.load_model(algo)
                        if model is None:
                            self.queue.put(f"Failed to load model for {algo}. Skipping...")
                            logging.error(f"Failed to load model for {algo}. Skipping...")
                            continue
                    except ValueError as ve:
                        self.queue.put(str(ve))
                        logging.error(str(ve))
                        continue

                    try:
                        ground_truths = self.load_ground_truths(sequence_folder)
                        self.queue.put(f"Loaded ground truth annotations from '{sequence_folder}'.")
                        logging.info(f"Loaded ground truth annotations from '{sequence_folder}'.")
                    except FileNotFoundError as fe:
                        self.queue.put(str(fe))
                        logging.error(str(fe))
                        continue

                    img_folder = os.path.join(sequence_folder, 'img1')
                    img_files = sorted(glob.glob(os.path.join(img_folder, '*.jpg')))

                    total_frames = len(img_files)
                    seqinfo_path = os.path.join(sequence_folder, 'seqinfo.ini')
                    if os.path.isfile(seqinfo_path):
                        config = configparser.ConfigParser()
                        config.read(seqinfo_path)
                        try:
                            fps = float(config['Sequence']['frameRate'])
                        except (KeyError, ValueError):
                            fps = 30.0
                            logging.warning(f"Failed to read frameRate from '{seqinfo_path}'. Using default FPS: {fps}")
                    else:
                        fps = 30.0
                        logging.warning(f"seqinfo.ini not found in '{sequence_folder}'. Using default FPS: {fps}")

                    if total_frames == 0:
                        self.queue.put(f"No images found in '{img_folder}'. Skipping sequence.")
                        logging.warning(f"No images found in '{img_folder}'. Skipping sequence.")
                        continue

                    calibration_size = min(50, total_frames)
                    calibration_frames = []
                    for idx in range(calibration_size):
                        img_file = img_files[idx]
                        frame = cv2.imread(img_file)
                        if frame is None:
                            self.queue.put(f"Unable to read image '{img_file}'. Skipping frame.")
                            logging.error(f"Unable to read image '{img_file}'. Skipping frame.")
                            continue
                        gt = ground_truths.get(idx, None)
                        if gt is None:
                            continue
                        calibration_frames.append((frame, gt))

                    if self.auto_threshold_var.get():
                        calibration_method = self.calibration_method_var.get()
                        self.queue.put(f"Starting calibration using {calibration_method} search...")
                        logging.info(f"Starting calibration using {calibration_method} search...")
                        best_threshold = self.calibrate_model(algo, model, calibration_frames, calibration_method=calibration_method)
                        model.set_confidence_threshold(best_threshold)
                        self.queue.put(f"Calibration completed. Set Confidence Threshold to {best_threshold:.2f}.")
                        logging.info(f"Calibration completed. Set Confidence Threshold to {best_threshold:.2f}.")
                    else:
                        initial_threshold = self.conf_threshold.get()
                        model.set_confidence_threshold(initial_threshold)
                        self.queue.put(f"Set Confidence Threshold to {initial_threshold:.2f}.")
                        logging.info(f"Set Confidence Threshold to {initial_threshold:.2f}.")

                    frame = cv2.imread(img_files[0])
                    if frame is None:
                        self.queue.put(f"Unable to read the first image in '{img_folder}'. Skipping sequence.")
                        logging.error(f"Unable to read the first image in '{img_folder}'. Skipping sequence.")
                        continue
                    height, width = frame.shape[:2]

                    output_subdir = os.path.join(output_dir, f"{sequence_name}_{algo}")
                    os.makedirs(output_subdir, exist_ok=True)

                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    output_video_path = os.path.join(output_subdir, f"{sequence_name}_{algo}_output.mp4")
                    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

                    if not out_video.isOpened():
                        self.queue.put(f"Failed to open video writer for '{output_video_path}'. Skipping sequence.")
                        logging.error(f"Failed to open video writer for '{output_video_path}'. Skipping sequence.")
                        continue

                    start_time = time.time()

                    map_metric = MeanAveragePrecision(iou_type="bbox").to(model.device)

                    total_confidence = 0.0
                    num_detections = 0

                    total_TP = 0
                    total_FP = 0
                    total_FN = 0

                    iou_threshold = 0.5
                    iou_sum = 0.0

                    batch_size = 4
                    num_batches = int(np.ceil(total_frames / batch_size))
                    logging.debug(f"Total frames: {total_frames}, Batch size: {batch_size}, Num batches: {num_batches}")

                    for batch_idx in range(num_batches):
                        if self.stop_event.is_set():
                            self.queue.put("Processing stopped by user.")
                            logging.info("Processing stopped by user.")
                            break

                        batch_start = batch_idx * batch_size
                        batch_end = min((batch_idx + 1) * batch_size, total_frames)
                        batch_frames = []
                        batch_indices = []
                        for idx in range(batch_start, batch_end):
                            img_file = img_files[idx]
                            frame = cv2.imread(img_file)
                            if frame is None:
                                self.queue.put(f"Unable to read image '{img_file}'. Skipping frame.")
                                logging.error(f"Unable to read image '{img_file}'. Skipping frame.")
                                continue
                            batch_frames.append(frame)
                            batch_indices.append(idx)

                        detections_batch = []
                        for frame in batch_frames:
                            detections = model.detect(frame)
                            if detections is None:
                                detections = {'boxes': [], 'scores': [], 'labels': []}
                            detections_batch.append(detections)

                        for idx, detections in zip(batch_indices, detections_batch):
                            current_frame_idx = idx + 1
                            self.queue.put(f"Processing frame {current_frame_idx}/{total_frames} ({algo})")
                            logging.info(f"Processing frame {current_frame_idx}/{total_frames} ({algo})")

                            ground_truth = ground_truths.get(idx, None)
                            if ground_truth is None:
                                continue

                            pred_labels = self.map_labels(algo, detections['labels'])
                            gt_labels = ground_truth['labels']

                            valid_preds = []
                            for box, score, label in zip(detections['boxes'], detections['scores'], pred_labels):
                                if label == 0:
                                    valid_preds.append({'boxes': box, 'scores': score, 'labels': label})

                            if len(valid_preds) == 0:
                                preds = [{
                                    'boxes': torch.empty((0, 4), dtype=torch.float32).to(model.device),
                                    'scores': torch.tensor([], dtype=torch.float32).to(model.device),
                                    'labels': torch.tensor([], dtype=torch.int64).to(model.device)
                                }]
                            else:
                                preds = [{
                                    'boxes': torch.tensor([item['boxes'] for item in valid_preds], dtype=torch.float32).to(model.device),
                                    'scores': torch.tensor([item['scores'] for item in valid_preds], dtype=torch.float32).to(model.device),
                                    'labels': torch.tensor([item['labels'] for item in valid_preds], dtype=torch.int64).to(model.device)
                                }]

                            targets = [{
                                'boxes': torch.tensor(ground_truth['boxes'], dtype=torch.float32).to(model.device),
                                'labels': torch.tensor(ground_truth['labels'], dtype=torch.int64).to(model.device)
                            }]

                            if not preds[0]['boxes'].shape[0]:
                                logging.warning(f"No predictions for frame {current_frame_idx}.")
                            if not targets[0]['boxes'].shape[0]:
                                logging.warning(f"No ground truth boxes for frame {current_frame_idx}.")

                            try:
                                map_metric.update(preds, targets)
                            except Exception as e:
                                logging.error(f"Error updating map_metric for frame {current_frame_idx}: {e}", exc_info=True)

                            total_confidence += torch.sum(preds[0]['scores']).item()
                            num_detections += len(preds[0]['scores'])

                            TP, FP, FN, avg_iou = self.compute_confusion_matrix(preds[0], targets[0], iou_threshold)
                            total_TP += TP
                            total_FP += FP
                            total_FN += FN
                            iou_sum += avg_iou * TP

                            annotated_frame = visualize_detections(
                                batch_frames[idx - batch_start],
                                preds[0],
                                model.class_names,
                                algo,
                                model.confidence_threshold
                            )

                            out_video.write(annotated_frame)
                            logging.debug(f"Frame {current_frame_idx} written to output video.")

                            processed_tasks += 1
                            overall_progress = (processed_tasks / total_tasks) * 100
                            self.progress['value'] = overall_progress
                            self.update_idletasks()

                    end_time = time.time()
                    total_processing_time = end_time - start_time
                    self.processing_time[algo] = total_processing_time

                    out_video.release()

                    try:
                        map_result = map_metric.compute()
                        mAP = map_result['map'].item()
                    except Exception as e:
                        logging.error(f"Error computing mAP for {algo} on {sequence_name}: {e}", exc_info=True)
                        mAP = 0.0

                    if (total_TP + total_FP) > 0:
                        precision = total_TP / (total_TP + total_FP)
                    else:
                        precision = 0.0

                    if (total_TP + total_FN) > 0:
                        recall = total_TP / (total_TP + total_FN)
                    else:
                        recall = 0.0

                    if (precision + recall) > 0:
                        f1_score = 2 * (precision * recall) / (precision + recall)
                    else:
                        f1_score = 0.0

                    iou = iou_sum / total_TP if total_TP > 0 else 0.0
                    avg_conf = total_confidence / num_detections if num_detections > 0 else 0.0

                    metrics = {
                        'mAP': mAP,
                        'Precision': precision,
                        'Recall': recall,
                        'F1 Score': f1_score,
                        'IoU': iou,
                        'Processing Time (s)': round(total_processing_time, 2),
                        'Average Confidence': round(avg_conf, 2),
                        'TP': total_TP,
                        'FP': total_FP,
                        'FN': total_FN
                    }

                    metrics_image_path = os.path.join(output_subdir, f"{sequence_name}_{algo}_metrics_table.png")
                    save_metrics_table(metrics, metrics_image_path)
                    self.queue.put(f"Metrics table saved at '{metrics_image_path}'.")
                    logging.info(f"Metrics table saved at '{metrics_image_path}'.")

                    cm_image_path = os.path.join(output_subdir, f"{sequence_name}_{algo}_confusion_matrix.png")
                    save_confusion_matrix_image(metrics, cm_image_path)
                    self.queue.put(f"Confusion Matrix image saved at '{cm_image_path}'.")
                    logging.info(f"Confusion Matrix image saved at '{cm_image_path}'.")

                    self.queue.put(f"Displaying evaluation results for {algo} on {sequence_name}...")
                    logging.info(f"Displaying evaluation results for {algo} on {sequence_name}...")
                    self.display_evaluation_results(sequence_name, algo, metrics)

                    self.queue.put(f"Total Processing Time for {algo} on {sequence_name}: {round(total_processing_time, 2)} seconds")
                    logging.info(f"Total Processing Time for {algo} on {sequence_name}: {round(total_processing_time, 2)} seconds")

        except Exception as e:
            logging.error(f"An error occurred during processing: {e}", exc_info=True)
            self.queue.put(f"An error occurred: {e}")
        finally:
            self.enable_ui()
            self.progress['value'] = 100

    def calibrate_model(self, algo, model, calibration_frames, calibration_method="linear"):
        thresholds = np.linspace(0.3, 0.7, 9)
        best_threshold = 0.5
        best_map = 0.0

        if calibration_method == "linear":
            for threshold in thresholds:
                model.set_confidence_threshold(threshold)
                self.queue.put(f"Testing confidence threshold: {threshold:.2f}")
                logging.info(f"Testing confidence threshold: {threshold:.2f}")

                args = [(model, frame, gt) for (frame, gt) in calibration_frames]
                with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                    results = pool.map(process_frame_for_calibration, args)

                map_total = 0.0
                valid_frames = 0
                for preds, targets in results:
                    if not preds or not targets:
                        continue
                    map_metric = MeanAveragePrecision(iou_type="bbox").to(model.device)
                    try:
                        map_metric.update(preds, targets)
                        map_result = map_metric.compute()
                        map_total += map_result['map'].item()
                        valid_frames += 1
                    except Exception as e:
                        logging.error(f"Error computing mAP during calibration: {e}", exc_info=True)
                        continue

                if valid_frames > 0:
                    average_map = map_total / valid_frames
                else:
                    average_map = 0.0

                logging.info(f"Threshold: {threshold:.2f}, Average mAP: {average_map:.4f}")

                if average_map > best_map:
                    best_map = average_map
                    best_threshold = threshold

        elif calibration_method == "binary":
            lower = 0
            upper = len(thresholds) - 1
            while lower <= upper:
                mid = (lower + upper) // 2
                threshold = thresholds[mid]
                model.set_confidence_threshold(threshold)
                self.queue.put(f"Testing confidence threshold: {threshold:.2f}")
                logging.info(f"Testing confidence threshold: {threshold:.2f}")

                args = [(model, frame, gt) for (frame, gt) in calibration_frames]
                with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                    results = pool.map(process_frame_for_calibration, args)

                map_total = 0.0
                valid_frames = 0
                for preds, targets in results:
                    if not preds or not targets:
                        continue
                    map_metric = MeanAveragePrecision(iou_type="bbox").to(model.device)
                    try:
                        map_metric.update(preds, targets)
                        map_result = map_metric.compute()
                        map_total += map_result['map'].item()
                        valid_frames += 1
                    except Exception as e:
                        logging.error(f"Error computing mAP during calibration: {e}", exc_info=True)
                        continue

                if valid_frames > 0:
                    average_map = map_total / valid_frames
                else:
                    average_map = 0.0

                logging.info(f"Threshold: {threshold:.2f}, Average mAP: {average_map:.4f}")

                if average_map > best_map:
                    best_map = average_map
                    best_threshold = threshold
                    lower = mid + 1
                else:
                    upper = mid - 1

        logging.info(f"Optimal Confidence Threshold found: {best_threshold:.2f} with mAP: {best_map:.4f}")
        self.queue.put(f"Optimal Confidence Threshold found: {best_threshold:.2f} with mAP: {best_map:.4f}")
        return best_threshold

    def compute_confusion_matrix(self, preds, targets, iou_threshold=0.5):
        TP = 0
        FP = 0
        FN = 0
        detected_gt = set()

        pred_boxes = preds['boxes'].cpu().numpy()
        pred_labels = preds['labels'].cpu().numpy()

        gt_boxes = targets['boxes'].cpu().numpy()
        gt_labels = targets['labels'].cpu().numpy()

        iou_sum = 0.0

        for pred_idx, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if gt_idx in detected_gt:
                    continue
                if pred_label != gt_label:
                    continue

                iou = self.calculate_iou(pred_box, gt_box)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_gt_idx >= 0:
                TP += 1
                detected_gt.add(best_gt_idx)
                iou_sum += best_iou
            else:
                FP += 1

        FN = len(gt_boxes) - len(detected_gt)
        avg_iou = iou_sum / TP if TP > 0 else 0.0

        return TP, FP, FN, avg_iou

    def calculate_iou(self, boxA, boxB):
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

    def stop_processing(self):
        self.stop_event.set()
        self.queue.put("Stop signal sent. Attempting to stop processing...")
        self.stop_button.config(state='disabled')
        self.start_button.config(state='normal')

    def load_model(self, algo):
        device = torch.device('cpu')
        try:
            if algo == "YOLOv8s":
                model = YOLOv8Model(device=device)
                model.class_names = ['person']
                logging.info("YOLOv8s model loaded successfully.")
                return model

            elif algo == "Faster R-CNN":
                model = FasterRCNNModel(device=device)
                model.class_names = ['person']
                logging.info("Faster R-CNN model loaded successfully.")
                return model

            elif algo == "SSD":
                model = SSD300Model(device=device)
                model.class_names = ['person']
                logging.info("SSD model loaded successfully.")
                return model

            else:
                logging.warning(f"Unknown Algorithm: {algo}")
                raise ValueError(f"Unsupported model: {algo}")
        except Exception as e:
            logging.error(f"Error loading model {algo}: {e}", exc_info=True)
            self.queue.put(f"Error loading model {algo}: {e}")
            return None

    def load_ground_truths(self, sequence_folder):
        gt_file = os.path.join(sequence_folder, 'gt', 'gt.txt')
        if not os.path.isfile(gt_file):
            raise FileNotFoundError(f"Ground truth file not found at '{gt_file}'.")

        gt_data = pd.read_csv(gt_file, header=None)
        num_columns = gt_data.shape[1]

        if num_columns == 10:
            gt_data.columns = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width',
                               'bb_height', 'conf', 'class', 'visibility', 'x']
        elif num_columns == 9:
            gt_data.columns = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width',
                               'bb_height', 'conf', 'class', 'visibility']
        else:
            logging.warning(f"Unexpected number of columns in gt.txt: {num_columns}")
            gt_data = gt_data.iloc[:, :9]
            gt_data.columns = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width',
                               'bb_height', 'conf', 'class', 'visibility']

        essential_columns = ['frame', 'bb_left', 'bb_top', 'bb_width', 'bb_height']
        gt_data.dropna(subset=essential_columns, inplace=True)

        gt_data[essential_columns] = gt_data[essential_columns].astype(float)

        if 'conf' in gt_data.columns:
            gt_data = gt_data[gt_data['conf'] == 1]

        # Correctly map 'person' class
        gt_data['class'] = 0  # Assuming 'person' is the only class and mapped to 0 in the application

        ground_truths = {}
        for frame_idx in gt_data['frame'].unique():
            frame_data = gt_data[gt_data['frame'] == frame_idx]
            boxes = frame_data[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            labels = frame_data['class'].tolist()
            ground_truths[int(frame_idx) - 1] = {'boxes': boxes.tolist(), 'labels': labels}

        logging.info(f"Loaded ground truths for {len(ground_truths)} frames.")
        self.queue.put(f"Loaded ground truths for {len(ground_truths)} frames.")
        return ground_truths

    def map_labels(self, algo, labels):
        mapped_labels = []
        for label in labels:
            if algo == "YOLOv8s":
                if label == 0:
                    mapped_labels.append(0)
                else:
                    mapped_labels.append(-1)
            elif algo == "Faster R-CNN":
                if label == 1:
                    mapped_labels.append(0)
                else:
                    mapped_labels.append(-1)
            elif algo == "SSD":
                mapped_labels.append(label)
            else:
                mapped_labels.append(-1)
        return mapped_labels

    def display_evaluation_results(self, sequence_name, algo, metrics):
        tab_name = f"{sequence_name} - {algo}"
        if tab_name in self.results_tabs:
            tab = self.results_tabs[tab_name]
            self.results_notebook.forget(tab)
            del self.results_tabs[tab_name]

        tab = ttk.Frame(self.results_notebook)
        self.results_tabs[tab_name] = tab
        self.results_notebook.add(tab, text=tab_name)

        tree = ttk.Treeview(tab, columns=("Metric", "Value"), show='headings')
        tree.heading("Metric", text="Metric")
        tree.heading("Value", text="Value")
        tree.column("Metric", width=200, anchor='center')
        tree.column("Value", width=300, anchor='center')
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        metrics_to_display = ['mAP', 'Precision', 'Recall', 'F1 Score', 'IoU',
                              'Processing Time (s)', 'Average Confidence',
                              'TP', 'FP', 'FN']
        formatted_metrics = self.compute_metrics_display(metrics)

        for metric in metrics_to_display:
            value = formatted_metrics.get(metric, "0.00")
            tree.insert('', tk.END, values=(metric, value))

    def compute_metrics_display(self, metrics):
        formatted_metrics = {}
        for key, value in metrics.items():
            if key in ['mAP', 'Precision', 'Recall', 'F1 Score', 'IoU']:
                formatted_metrics[key] = f"{value * 100:.2f}%"
            elif key == 'Average Confidence':
                formatted_metrics[key] = f"{value:.2f}"
            elif key == 'Processing Time (s)':
                formatted_metrics[key] = f"{value:.2f} s"
            else:
                formatted_metrics[key] = f"{value}"
        return formatted_metrics

    def enable_ui(self):
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.sequence_listbox.config(state='normal')
        self.auto_threshold_checkbox.config(state='normal')
        self.toggle_threshold_entry()

    def disable_ui(self):
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.sequence_listbox.config(state='disabled')
        self.auto_threshold_checkbox.config(state='disabled')
        self.conf_threshold_entry.config(state='disabled')
        self.linear_search_radio.config(state='disabled')
        self.binary_search_radio.config(state='disabled')

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    app = ObjectRecognitionApp()


    app.mainloop()
