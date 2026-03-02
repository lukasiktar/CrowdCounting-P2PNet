import cv2
from pytubefix import YouTube
import os

# YouTube video URL
url = "https://www.youtube.com/watch?v=1hYWLZLgrR8"

# Download the video
yt = YouTube(url)
stream = yt.streams.filter(file_extension='mp4').first()
video_path = stream.download(output_path=".", filename="downloaded_video")

# Open video with OpenCV
cap = cv2.VideoCapture(video_path)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Read and display frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('YouTube Video', frame)
    out.write(frame)  # Store the frame
    if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' to exit
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Optionally, remove the downloaded file if you don't need it
os.remove(video_path)
