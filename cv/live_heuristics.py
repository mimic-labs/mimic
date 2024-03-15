import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

centers = [[], []]

# Function to calculate standard deviation of distances
def calculate_stdev(coordinates):
    distances = np.linalg.norm(coordinates - np.mean(coordinates, axis=0), axis=1)
    stdev = np.std(distances)
    return stdev

# Function to draw a circle on the frame
def draw_circle(frame, center):
    cv2.circle(frame, center, 20, (0, 255, 0), -1)
    
def extract_hand_data(frame):
    results = hands.process(frame)
    print(results.multi_hand_world_landmarks)

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
            # orientations[hand_idx].append(normal_vector)

            # Get hand center
            palm_points_mean = np.mean(palm_points, axis=0)
            center_x = int(palm_points_mean[0] * frame.shape[1])
            center_y = int(palm_points_mean[1] * frame.shape[0])
            centers[hand_idx].append((center_x, center_y))

            # Draw current & past hand centers on existing frame
            cv2.circle(frame, (center_x, center_y), 3, (255, 0, 0))
            cv2.polylines(frame, [np.array(centers[hand_idx])], False, (0,0,255), 2)
        
    return frame

# Open video capture
cap = cv2.VideoCapture('cv/test_vids/test_hand_vid.mov')

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Parameters
window_size = 20
stdev_threshold = 10
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Initialize variables
centers_window = []

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand center coordinates
    frame = extract_hand_data(frame)
    print(centers)
    hand_center = centers[0][-1] if len(centers[0]) > 0 else None
    print(hand_center)

    if hand_center is not None:
        # Add hand center to the window
        centers_window.append(hand_center)

        # Keep the window size limited to the last 20 frames
        if len(centers_window) > window_size:
            centers_window.pop(0)

        # Calculate standard deviation of distances
        if len(centers_window) == window_size:
            stdev = calculate_stdev(np.array(centers_window))

            # Check if stdev is below the threshold
            if stdev < stdev_threshold:
                # Draw a circle on the frame at the average center
                average_center = tuple(np.mean(centers_window, axis=0).astype(int))
                draw_circle(frame, average_center)

    # Display the frame
    cv2.imshow('Hand Tracking', frame)

    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture
cap.release()
cv2.destroyAllWindows()