from transformers import pipeline
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt


checkpoint = "Intel/dpt-large"
depth_estimator = pipeline("depth-estimation", model=checkpoint)

video_url = "cv/test_videos/IMG_8290.mov"   
cap = cv2.VideoCapture(video_url)

fps = cap.get(cv2.CAP_PROP_FPS)

interval_seconds = 10
interval_frames = int(fps * interval_seconds)
framesSaved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Video Frame", frame)
    
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % interval_frames == 0:
        framesSaved += 1
        predictions = depth_estimator(image)

        print(frame.shape)
        print("Frame at " + str(interval_seconds * framesSaved) + " seconds:")
        depth_array = predictions["predicted_depth"]
        depth_array = depth_array.numpy()[0]

        min_value = np.min(depth_array)
        max_value = np.max(depth_array)

        # Display the matrix as an image in grayscale
        plt.imshow(depth_array, cmap='gray', vmin=min_value, vmax=max_value, interpolation='nearest')

        # Add a colorbar to show the mapping of values to colors
        plt.colorbar()

        # Show the plot
        plt.show()
        

        print(depth_array, depth_array.shape)
        print(predictions["depth"].size)
        predictions["depth"].show()

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()