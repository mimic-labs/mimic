import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

centers = []
orientations = []

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Open a video capture
cap = cv2.VideoCapture("cv/test_videos/IMG_8290.MOV")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(frame_rgb)

    if results.multi_hand_world_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            palm_points = np.asarray([[hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z], 
                                      [hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y, hand_landmarks.landmark[5].z], 
                                      [hand_landmarks.landmark[17].x, hand_landmarks.landmark[17].y, hand_landmarks.landmark[17].z]])

            normal_vector = np.cross(palm_points[2] - palm_points[0], palm_points[1] - palm_points[2])
            normal_vector /= np.linalg.norm(normal_vector)
            orientations.append(normal_vector)

            # print(normal_vector)

            # Get hand center
            palm_points_mean = np.mean(palm_points, axis=0)

            center_x = int(palm_points_mean[0] * frame.shape[1])
            center_y = int(palm_points_mean[1] * frame.shape[0])

            centers.append((center_x, center_y))

            cv2.circle(frame, (center_x, center_y), 3, (255, 0, 0))
            cv2.polylines(frame, [np.array(centers)], False, (0,0,255), 2)

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27: 
        break

cv2.destroyAllWindows()

# Plot the vectors as lines starting at the points
for i in range(len(centers)):
    x, y = centers[i]
    dx, dy, dz = orientations[i]
    ax.quiver(x, y, 1, dx, dy, dz, color='b', length=0.01)

# Draw lines connecting the points
for i in range(len(centers) - 1):
    x = [centers[i][0], centers[i + 1][0]]
    y = [centers[i][1], centers[i + 1][1]]
    ax.plot(x, y, 1, color='r')

plt.show()

