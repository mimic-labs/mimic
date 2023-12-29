import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse
import pathlib
import itertools
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

# Process a video and return the hand centers and orientations
def process_video(video_path: pathlib.Path, output_dir: pathlib.Path):
    centers = [[], []]
    orientations = [[], []]

    # Open a video capture
    vid_path_str = video_path.resolve().as_posix()
    cap = cv2.VideoCapture(vid_path_str)
    num_frames = 0
    
    # Define video writer config
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)
    
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    
    out_name = f"{output_dir / video_path.stem}_out.mp4"
    print(f"Output path: {out_name}")
    
    # Initialize video writer
    out = cv2.VideoWriter(out_name, fourcc, 20.0, size)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        num_frames += 1

        if num_frames > FRAME_LIMIT: # limit number of frames to process
            break

        # Process the frame with MediaPipe Hands
        results = hands.process(frame_rgb)

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

                # Draw current & past hand centers
                cv2.circle(frame, (center_x, center_y), 3, (255, 0, 0))
                cv2.polylines(frame, [np.array(centers[hand_idx])], False, (0,0,255), 2)

        cv2.imshow("Hand Tracking", frame) # show frame
        out.write(frame) # write frame to output video

        if cv2.waitKey(1) & 0xFF == 27: 
            break

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
def save_data(data, csv_path):
    # Alternatively can use `np.savetxt(csv_path, data, delimiter=",", fmt="%s")`?
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

# Initialize argument parser
def init_arg_parser():
    parser = argparse.ArgumentParser(description='Hand Path Extraction')
    parser.add_argument('-i', '--input-video-dir', type=str, default='test_videos',
                        help='Directory of input videos to process')
    parser.add_argument('-d', '--output-csv-path', type=str, default='../train_data/test.csv',
                        help='Path of CSV file to save output to')
    parser.add_argument('-o', '--output-video-dir', type=str, default='output',
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
        
    save_data(hand_data, output_csv_path)
    
    print(f"Hand data from {len(hand_data)} videos saved to {output_csv_path}")