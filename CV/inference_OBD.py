import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel
import warnings
import os
import json
import cv2

import pyrealsense2 as rs

warnings.filterwarnings("ignore")

# Fix SDP bug for Windows
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# Default paths rewritten to be relative for GitHub use
DEFAULT_IMAGE_PATH = "Florence2/test_images/overlap_NIST.jpg"
DEFAULT_MODEL_PATH = "Florence2/Saved_Models/epoch_12"

def load_florence2_model(model_path=None, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Loading Florence2 model on {device}...")

    # Load base Florence-2 model
    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True)

    # Load LoRA adapter if path is provided
    if model_path and os.path.exists(model_path):
        print(f"[INFO] Loading LoRA adapter from {model_path}...")
        model = PeftModel.from_pretrained(model, model_path).eval()
        print("[INFO] LoRA adapter loaded successfully.")
    else:
        print("[INFO] No LoRA adapter path provided or path doesn't exist. Using base model.")
        model = model.eval()

    return model, processor, device


def detect_objects_florence2(image, model, processor, device, task_prompt="<OD>"):
    print("[INFO] Running Florence2 object detection...")

    prompt = task_prompt
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))

    result = parsed_answer.get(task_prompt, {})

    # Normalize format: handle both 'labels' and 'bboxes_labels' keys
    if 'bboxes_labels' in result and 'labels' not in result:
        result['labels'] = result.pop('bboxes_labels')

    return result


def filter_target_object(detection_results, target_object):
    if not detection_results or 'bboxes' not in detection_results:
        return {'bboxes': [], 'labels': []}

    filtered_bboxes = []
    filtered_labels = []

    for bbox, label in zip(detection_results['bboxes'], detection_results['labels']):
        if target_object.lower() in label.lower() or label.lower() in target_object.lower():
            filtered_bboxes.append(bbox)
            filtered_labels.append(label)

    return {'bboxes': filtered_bboxes, 'labels': filtered_labels}


def filter_obstacles(detection_results, target_object):
    if not detection_results or 'bboxes' not in detection_results:
        return {'bboxes': [], 'labels': []}

    obstacle_bboxes = []
    obstacle_labels = []

    for bbox, label in zip(detection_results['bboxes'], detection_results['labels']):
        # Check if this is NOT the target object
        if not (target_object.lower() in label.lower() or label.lower() in target_object.lower()):
            obstacle_bboxes.append(bbox)
            obstacle_labels.append('obstacle')

    return {'bboxes': obstacle_bboxes, 'labels': obstacle_labels}


def plot_detections(image, target_results, obstacle_results, target_object):
    if not isinstance(image, np.ndarray):
        img_array = np.array(image).copy()
    else:
        img_array = image.copy()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img_array)

    # Mask obstacles with semi-transparent red overlay (so objects are still visible)
    if obstacle_results and 'bboxes' in obstacle_results and len(obstacle_results['bboxes']) > 0:
        for bbox in obstacle_results['bboxes']:
            x1, y1, x2, y2 = bbox

            # Draw semi-transparent red overlay on obstacle regions
            rect_mask = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=0,
                edgecolor='none',
                facecolor='red',
                alpha=0.3
            )
            ax.add_patch(rect_mask)

    # Draw obstacle bounding boxes and labels
    if obstacle_results and 'bboxes' in obstacle_results and len(obstacle_results['bboxes']) > 0:
        for bbox, label in zip(obstacle_results['bboxes'], obstacle_results['labels']):
            x1, y1, x2, y2 = bbox

            # Draw obstacle bounding box (red) - on top of the overlay
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor='red',
                facecolor='none',
                linestyle='--'
            )
            ax.add_patch(rect)

            # Add obstacle label
            ax.text(
                x1,
                y1 - 10,
                label,
                color='white',
                fontsize=10,
                fontweight='bold',
                bbox=dict(facecolor='red', alpha=0.7, edgecolor='black', linewidth=1)
            )

    # Draw target object bounding boxes and center points
    centers = []
    if target_results and 'bboxes' in target_results and len(target_results['bboxes']) > 0:
        for bbox, label in zip(target_results['bboxes'], target_results['labels']):
            x1, y1, x2, y2 = bbox

            # Calculate center point
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            centers.append((cx, cy))

            # Draw target bounding box (green)
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=3,
                edgecolor='lime',
                facecolor='none'
            )
            ax.add_patch(rect)

            # Draw center point as green dot
            ax.plot(
                cx,
                cy,
                'go',
                markersize=10,
                markeredgecolor='darkgreen',
                markeredgewidth=2,
                label='Center' if len(centers) == 1 else ''
            )

            # Add target label
            ax.text(
                x1,
                y1 - 10,
                label,
                color='black',
                fontsize=12,
                fontweight='bold',
                bbox=dict(facecolor='lime', alpha=0.8, edgecolor='black', linewidth=1)
            )

    # Set title
    target_count = len(target_results['bboxes']) if target_results and 'bboxes' in target_results else 0
    obstacle_count = len(obstacle_results['bboxes']) if obstacle_results and 'bboxes' in obstacle_results else 0
    ax.set_title(
        f"Target: '{target_object}' ({target_count} instance(s)) | Obstacles: {obstacle_count}",
        fontsize=16,
        fontweight='bold'
    )
    ax.axis('off')
    plt.tight_layout()
    plt.show()

    if target_count > 0:
        print(f"[SUCCESS] Found {target_count} instance(s) of '{target_object}'")
    if obstacle_count > 0:
        print(f"[INFO] Masked {obstacle_count} obstacle(s)")


# ===================== RealSense LIVE MODE =====================

import uuid
from collections import defaultdict

def iou(box1, box2):
    """Compute Intersection over Union between two boxes."""
    x1, y1, x2, y2 = box1
    X1, Y1, X2, Y2 = box2

    inter_x1 = max(x1, X1)
    inter_y1 = max(y1, Y1)
    inter_x2 = min(x2, X2)
    inter_y2 = min(y2, Y2)

    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (X2 - X1) * (Y2 - Y1)

    return inter_area / float(box1_area + box2_area - inter_area)


def assign_ids(existing_objects, new_boxes, iou_threshold=0.5):
    """
    Assign identity to each detected object using IOU-based tracking.
    existing_objects = dict(id → list_of_bboxes)
    new_boxes = list of bboxes detected in the frame
    """
    id_assignments = {}
    used_ids = set()

    for box in new_boxes:
        best_id = None
        best_iou = 0.0

        for obj_id, prev_boxes in existing_objects.items():
            last_box = prev_boxes[-1]
            score = iou(box, last_box)

            if score > best_iou and score >= iou_threshold and obj_id not in used_ids:
                best_iou = score
                best_id = obj_id

        if best_id is None:
            # Create a new id
            best_id = f"obstacle_{uuid.uuid4().hex[:6]}"

        id_assignments[best_id] = box
        used_ids.add(best_id)

    return id_assignments


def run_realsense_detection(target_object, model, processor, device, num_frames=10):

    print("[INFO] Starting RealSense pipeline...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    frame_count = 0

    # STORAGE FOR TRACKING
    target_boxes = []
    target_frame_hits = 0

    obstacle_tracks = defaultdict(list)
    obstacle_frame_hits = defaultdict(int)

    try:
        while frame_count < num_frames:

            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert to numpy
            rgb_bgr = np.asanyarray(color_frame.get_data())   # BGR for cv2
            rgb = rgb_bgr[:, :, ::-1]                         # RGB for Florence2
            image_pil = Image.fromarray(rgb)

            # RUN OBJECT DETECTION
            detection = detect_objects_florence2(image_pil, model, processor, device)
            target_det = filter_target_object(detection, target_object)
            obstacle_det = filter_obstacles(detection, target_object)

            # DRAW ON IMAGE (OpenCV)

            # Draw TARGET boxes
            if target_det["bboxes"]:
                target_frame_hits += 1
                for box in target_det["bboxes"]:
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    target_boxes.append(box)

                    # Green box
                    cv2.rectangle(rgb_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Green center point
                    cv2.circle(rgb_bgr, (cx, cy), 5, (0, 255, 0), -1)

                    # Label
                    cv2.putText(
                        rgb_bgr,
                        f"{target_object}",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

            # Draw OBSTACLES
            if obstacle_det["bboxes"]:
                # Assign obstacle IDs by IOU tracking
                assigned = assign_ids(obstacle_tracks, obstacle_det["bboxes"])

                for obj_id, box in assigned.items():
                    obstacle_tracks[obj_id].append(box)
                    obstacle_frame_hits[obj_id] += 1

                    x1, y1, x2, y2 = map(int, box)

                    # Red box
                    cv2.rectangle(rgb_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    # Label text
                    cv2.putText(
                        rgb_bgr,
                        obj_id,
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

            # Show progress
            cv2.putText(
                rgb_bgr,
                f"Frame {frame_count + 1}/{num_frames}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            cv2.imshow("Florence2 RealSense Detection", rgb_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Interrupted by user.")
                break

            frame_count += 1

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    # FRAME VALIDATION (≥50% detection rule)
    min_required = num_frames * 0.5

    target_valid = target_frame_hits >= min_required
    valid_obstacles = {
        obj_id: boxes
        for obj_id, boxes in obstacle_tracks.items()
        if obstacle_frame_hits[obj_id] >= min_required
    }

    # AVERAGING FUNCTION
    def avg_box(boxes):
        xs1 = [b[0] for b in boxes]
        ys1 = [b[1] for b in boxes]
        xs2 = [b[2] for b in boxes]
        ys2 = [b[3] for b in boxes]
        return {
            "x1": float(np.mean(xs1)),
            "y1": float(np.mean(ys1)),
            "x2": float(np.mean(xs2)),
            "y2": float(np.mean(ys2)),
        }

    # SAVE JSON OUTPUT (relative path within CV/)
    results = {
        "meta": {
            "frames": num_frames,
            "required_hits": min_required
        },
        "target": {},
        "obstacles": []
    }

    if target_valid and target_boxes:
        results["target"] = {
            "avg_bbox": avg_box(target_boxes),
            "frames_detected": int(target_frame_hits)
        }
    else:
        results["target"] = "Target not confidently detected"

    for obj_id, boxes in valid_obstacles.items():
        results["obstacles"].append({
            "id": obj_id,
            "avg_bbox": avg_box(boxes),
            "frames_detected": int(obstacle_frame_hits[obj_id])
        })

    with open("CV/objects_position.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n[SAVED] objects_position.json")
    print(json.dumps(results, indent=4))


def main():
    parser = argparse.ArgumentParser(
        description='Detect target object using Florence2 (image or RealSense live).'
    )
    parser.add_argument(
        '--image_path',
        type=str,
        default=DEFAULT_IMAGE_PATH,
        help=f'Path to the input image (default: {DEFAULT_IMAGE_PATH})'
    )
    parser.add_argument(
        '--target_object',
        type=str,
        default=None,
        help='Name of the target object to detect'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f'Path to Florence2 LoRA adapter checkpoint (default: {DEFAULT_MODEL_PATH})'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to run inference on (cuda/cpu). Auto-detected if not specified.'
    )
    parser.add_argument(
        '--live_cam',
        action='store_true',
        help='Use live RealSense camera for the first 10 frames instead of a static image.'
    )

    args = parser.parse_args()

    # Interactive mode for target object if not provided
    if args.target_object is None:
        args.target_object = input(
            "Enter the target object name to detect: "
        ).strip()

    # Load Florence2 model once
    model, processor, device = load_florence2_model(args.model_path, args.device)

    # ---- RealSense live mode ----
    if args.live_cam:
        print("[INFO] Running in RealSense LIVE mode for first 10 frames...")
        run_realsense_detection(
            target_object=args.target_object,
            model=model,
            processor=processor,
            device=device,
            num_frames=10
        )
        return

    # ---- Static image mode ----
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"[ERROR] Image not found at: {args.image_path}")
        return

    # Load image
    print(f"[INFO] Loading image from: {args.image_path}")
    image = Image.open(args.image_path).convert("RGB")
    print(f"[INFO] Image size: {image.size}")

    # Detect objects
    detection_results = detect_objects_florence2(image, model, processor, device)

    print(f"\n[INFO] All detections: {detection_results}")

    # Filter for target object and obstacles
    target_results = filter_target_object(detection_results, args.target_object)
    obstacle_results = filter_obstacles(detection_results, args.target_object)

    # Display results with obstacles masked
    plot_detections(image, target_results, obstacle_results, args.target_object)

    # Print summary
    print(f"\n[SUMMARY]")
    if target_results['bboxes']:
        print(f"  Target object: {args.target_object}")
        print(f"  Instances found: {len(target_results['bboxes'])}")
        for i, (bbox, label) in enumerate(
            zip(target_results['bboxes'], target_results['labels']), 1
        ):
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            print(f"  Instance {i}: {label}")
            print(f"    Bounding box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
            print(f"    Center point: ({cx:.1f}, {cy:.1f})")
    else:
        print(f"  No instances of '{args.target_object}' found in the image.")

    if obstacle_results['bboxes']:
        print(f"\n  Obstacles detected: {len(obstacle_results['bboxes'])}")
        for i, bbox in enumerate(obstacle_results['bboxes'], 1):
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            print(f"  Obstacle {i}:")
            print(f"    Bounding box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
            print(f"    Center point: ({cx:.1f}, {cy:.1f})")


if __name__ == "__main__":
    main()

