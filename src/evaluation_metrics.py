import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

def parse_yolo_label(file_path):
    """Parses a YOLO format label file."""
    boxes = []
    with open(file_path, 'r') as f:
        for line in f:
            class_id, cx, cy, w, h = map(float, line.strip().split())
            boxes.append((int(class_id), cx, cy, w, h))
    return boxes

def bbox_iou(box1, box2):
    """Computes IoU (Intersection over Union) between two bounding boxes."""
    _, x1, y1, w1, h1 = box1
    _, x2, y2, w2, h2 = box2

    x1_min, y1_min = x1 - w1 / 2, y1 - h1 / 2
    x1_max, y1_max = x1 + w1 / 2, y1 + h1 / 2

    x2_min, y2_min = x2 - w2 / 2, y2 - h2 / 2
    x2_max, y2_max = x2 + w2 / 2, y2 + h2 / 2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def match_boxes(gt_boxes, pred_boxes, iou_threshold=0.5):
    """Matches ground-truth boxes to predicted boxes based on IoU."""
    matches = []
    unmatched_preds = set(range(len(pred_boxes)))

    for gt_idx, gt_box in enumerate(gt_boxes):
        best_match = -1
        best_iou = iou_threshold

        for pred_idx in list(unmatched_preds):
            iou = bbox_iou(gt_box, pred_boxes[pred_idx])
            if iou > best_iou:
                best_iou = iou
                best_match = pred_idx

        if best_match >= 0:
            matches.append((gt_idx, best_match))
            unmatched_preds.remove(best_match)

    return matches, unmatched_preds

def evaluate_metrics(gt_boxes, pred_boxes, iou_threshold=0.5):
    """Calculates precision, recall, and F1 score based on matches."""
    matches, unmatched_preds = match_boxes(gt_boxes, pred_boxes, iou_threshold)

    tp = len(matches)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1_score

def calculate_map(gt_boxes_list, pred_boxes_list, iou_threshold=0.5):
    """Calculates mean Average Precision (mAP)."""
    all_precisions = []

    for gt_boxes, pred_boxes in zip(gt_boxes_list, pred_boxes_list):
        precision, _, _ = evaluate_metrics(gt_boxes, pred_boxes, iou_threshold)
        all_precisions.append(precision)

    return np.mean(all_precisions) if all_precisions else 0

def calculate_overall_metrics(gt_boxes_list, pred_boxes_list, iou_threshold=0.5):
    """Calculates overall precision, recall, and F1 score."""
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for gt_boxes, pred_boxes in zip(gt_boxes_list, pred_boxes_list):
        matches, unmatched_preds = match_boxes(gt_boxes, pred_boxes, iou_threshold)
        tp = len(matches)
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp

        total_tp += tp
        total_fp += fp
        total_fn += fn

    overall_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
    overall_f1_score = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if overall_precision + overall_recall > 0 else 0

    return overall_precision, overall_recall, overall_f1_score

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <original_labels_folder> <predicted_labels_folder>")
        sys.exit(1)

    gt_folder = Path(sys.argv[1])
    pred_folder = Path(sys.argv[2])

    gt_files = {f.name: f for f in gt_folder.glob("*.txt")}
    pred_files = {f.name: f for f in pred_folder.glob("*.txt")}

    all_gt_boxes = []
    all_pred_boxes = []

    for file_name, gt_file in gt_files.items():
        if file_name not in pred_files:
            print(f"Warning: No prediction file for {file_name}")
            continue

        pred_file = pred_files[file_name]

        gt_boxes = parse_yolo_label(gt_file)
        pred_boxes = parse_yolo_label(pred_file)

        all_gt_boxes.append(gt_boxes)
        all_pred_boxes.append(pred_boxes)

        precision, recall, f1_score = evaluate_metrics(gt_boxes, pred_boxes)
        print(f"File: {file_name} | Precision: {precision:.2f} | Recall: {recall:.2f} | F1 Score: {f1_score:.2f}")

    map_score = calculate_map(all_gt_boxes, all_pred_boxes)
    overall_precision, overall_recall, overall_f1_score = calculate_overall_metrics(all_gt_boxes, all_pred_boxes)

    print(f"Overall mAP: {map_score:.2f}")
    print(f"Overall Precision: {overall_precision:.2f}")
    print(f"Overall Recall: {overall_recall:.2f}")
    print(f"Overall F1 Score: {overall_f1_score:.2f}")

if __name__ == "__main__":
    main()
