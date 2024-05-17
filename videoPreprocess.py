import cv2

def preprocess_video(video_path, preprocess_image):
    # Open video
    video = cv2.VideoCapture(video_path)
    frames = []
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        # Preprocess frame as image
        frame = preprocess_image(frame)
        frames.append(frame)
    video.release()
    return frames 