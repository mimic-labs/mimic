import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse
import pathlib
import itertools
import os

DETIC_DIR = pathlib.Path.cwd() / "third_party/Detic"

os.chdir(DETIC_DIR) # needed because Detic code contains hardcoded relative paths

from detectron2.config import get_cfg
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.predictor import VisualizationDemo

# Uncomment if Python package paths are fixed
# from third_party.Detic.demo import test_opencv_video_format, setup_cfg

# import logging

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

FRAME_LIMIT = 200   # max number of frames to process for each video
VIDEO_LIMIT = 10    # max number of videos to process

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initializes necessary config from command line args. Copied from third_party/Detic/demo.py
# Delete once Python package paths are fixed & imports work
def setup_cfg(args):
    cfg = get_cfg()
    
    if args.cpu:
        cfg.MODEL.DEVICE="cpu"
        
    add_centernet_config(cfg)
    add_detic_config(cfg)
    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later

    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    
    cfg.freeze()
    
    return cfg

def init_video_writer(cap, output_name):
    # Define video writer config
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)

    frames_per_second = cap.get(cv2.CAP_PROP_FPS)
    
    # Uncomment if Python package paths are fixed & imports work
    # codec, file_ext = (
    #     ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
    # )
    codec, file_ext = ("mp4v", ".mp4") # hardcoded to mp4 for now
    fourcc = cv2.VideoWriter_fourcc(*codec)
    
    output_path = f"{output_name}{file_ext}"
    print(f"Output path: {output_path}")
    
    return cv2.VideoWriter(
        filename=output_path,
        # some installation of opencv may not support x264 (due to its license),
        # you can try other format (e.g. MPEG)
        fourcc=fourcc,
        fps=float(frames_per_second),
        frameSize=size,
        isColor=True,
    )

# Initialize the Detic predictor (visualization demo)
def init_detic_predictor(vocab=None):
    # Namespace(config_file='configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml', webcam=None, cpu=False, video_input='..\\..\\cv\\output_vids\\52bae57c-0f27-45ff-892f-bdc87ee27ea9_out.mp4', input=None, output='..\\..\\cv\\output_vids\\cooking_combined2.mp4', vocabulary='custom', custom_vocabulary='bowl,chopsticks,human,mug,sauce bottle,plate,glass,tomatoes', pred_all_class=False, confidence_threshold=0.3, opts=['MODEL.WEIGHTS', 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'])
    vocab = 'bowl,chopsticks,human,mug,sauce bottle,plate,glass,tomatoes'

    config_dict = {
        'config_file': 'configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml',
        'cpu': False,
        'video_input': '',
        'output': '',
        'vocabulary': 'custom',
        'custom_vocabulary': vocab,
        'pred_all_class': False,
        'confidence_threshold': 0.3,
        'opts': ['MODEL.WEIGHTS', 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth']
    }
    
    args = argparse.Namespace(**config_dict) # convert hardcoded dict to parsed args
    cfg = setup_cfg(args)

    return VisualizationDemo(cfg, args) # parallel = True doesn't work since os.chdir shouldn't run multiple times

# Updates centers & orientations with current frame's hand data, returns annotated frame
def extract_hand_data(centers, orientations, frame):
    results = hands.process(frame)

    if results.multi_hand_world_landmarks:
        for hand in results.multi_handedness:
            # Get a constant index for the detected hand (0 or 1). If only 1 hand is detected, default to index = 0.
            hand_idx = hand.classification[0].index if len(results.multi_hand_landmarks) > 1 else 0
            hand_landmarks = results.multi_hand_landmarks[hand_idx]
            
            # Get key points on palm
            palm_points = np.asarray([[hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z], 
                                    [hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y, hand_landmarks.landmark[5].z], 
                                    [hand_landmarks.landmark[17].x, hand_landmarks.landmark[17].y, hand_landmarks.landmark[17].z]])

            # Get palm orientation by calculating normal vector of palm plane
            normal_vector = np.cross(palm_points[2] - palm_points[0], palm_points[1] - palm_points[2])
            normal_vector /= np.linalg.norm(normal_vector)
            orientations[hand_idx].append(normal_vector)

            # Get hand center
            palm_points_mean = np.mean(palm_points, axis=0)
            center_x = int(palm_points_mean[0] * frame.shape[1])
            center_y = int(palm_points_mean[1] * frame.shape[0])
            centers[hand_idx].append((center_x, center_y))

            # Draw current & past hand centers on existing frame
            cv2.circle(frame, (center_x, center_y), 3, (255, 0, 0))
            cv2.polylines(frame, [np.array(centers[hand_idx])], False, (0,0,255), 2)
    
    return frame

def object_hand_distance(hand_coords, object_coords):
    pass

# Process a video using 1) hand paths, and 2) segmented objects. Return just the hand centers and orientations for now.
def process_video(video_path: pathlib.Path, output_dir: pathlib.Path = None):
    centers = [[], []]
    orientations = [[], []]

    # Open a video capture
    vid_path_str = video_path.resolve().as_posix()
    cap = cv2.VideoCapture(vid_path_str)
    num_frames = 0
    
    if output_dir is not None:
        # Set up video writer
        output_name = f"{output_dir / video_path.stem}_out"
        out = init_video_writer(cap, output_name)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    
    # Initialize Detic predictor for instance segmentation
    detic_predictor = init_detic_predictor()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        num_frames += 1

        if num_frames > FRAME_LIMIT: # limit number of frames to process
            break
        
        hand_frame = extract_hand_data(centers, orientations, frame_rgb) # updates centers & orientations, returns annotated frame
        
        predictions, visualized_output = detic_predictor.run_on_image(hand_frame) # performs instance segmentation on annotated frame
        output_img = visualized_output.get_image()[:, :, ::-1]

        instances = predictions["instances"].get_fields()
        obj_boxes = instances['pred_boxes'].tensor.cpu().numpy()
        obj_classes = instances['pred_classes'].tensor.cpu().numpy()
        # print(predictions)
        
        cv2.imshow("Hand Tracking", output_img) # show the frame to user
        if output_dir is not None:
            out.write(output_img) # save frame to output video

        if cv2.waitKey(1) in (27, ord('q')): # esc or q to quit
            break
    
    cap.release()
    out.release()

    cv2.destroyAllWindows()

    return centers, orientations

# Plot the hand paths & orientations
def plot_hand_paths(centers, orientations):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for centers_list, orientations_list in zip(centers, orientations):
        # Plot the vectors as lines starting at the points
        for i in range(len(centers_list)):
            x, y = centers_list[i]
            dx, dy, dz = orientations_list[i]
            ax.quiver(x, y, 1, dx, dy, dz, color='b', length=0.01)

        # Draw lines connecting the points
        for i in range(len(centers_list) - 1):
            x = [centers_list[i][0], centers_list[i + 1][0]]
            y = [centers_list[i][1], centers_list[i + 1][1]]
            ax.plot(x, y, 1, color='r')

    plt.show()

# Save (centers, orientations) hand data to CSV file. 
def save_data(data, csv_path: pathlib.Path):
    # Alternatively can use `np.savetxt(csv_path, data, delimiter=",", fmt="%s")`?
    pathlib.Path(csv_path.parent).mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

# Initialize argument parser
def init_arg_parser():
    # All paths must be relative to third_party/Detic due to the os.chdir call at the top
    parser = argparse.ArgumentParser(description='Hand Path Extraction')
    parser.add_argument('-i', '--input-video-dir', type=str, default='../../cv/input_vids',
                        help='Directory of input videos to process')
    parser.add_argument('-d', '--output-csv-path', type=str, default='../../datasets/train_data.csv',
                        help='Path of CSV file to save output to')
    parser.add_argument('-o', '--output-video-dir', type=str, default='../../cv/output_vids',
                    help='Directory of output videos to save to')
    return parser

if __name__ == "__main__":
    hand_data = []

    parser = init_arg_parser()
    args = parser.parse_args()

    input_vid_dir = pathlib.Path(args.input_video_dir)
    output_csv_path = pathlib.Path(args.output_csv_path)
    output_vid_dir = pathlib.Path(args.output_video_dir)
    
    print(f"Input video directory: {input_vid_dir}")
    print(f"{len(list(input_vid_dir.rglob('*')))} total videos found: {list(input_vid_dir.rglob('*'))}")

    # Recursively process all videos in video input directory, up to VIDEO_LIMIT videos
    for vid_path in itertools.islice(input_vid_dir.rglob('*'), VIDEO_LIMIT):
        centers, orientations = process_video(vid_path, output_vid_dir)
        hand_data.append(list(zip(centers, orientations)))
        plot_hand_paths(centers, orientations)
        
    # save_data(hand_data, output_csv_path)
    
    print(f"Hand data from {len(hand_data)} videos saved to {output_csv_path}")