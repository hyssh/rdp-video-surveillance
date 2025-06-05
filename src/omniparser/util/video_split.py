# Read a video file and split it into frames
# Save the frames to a directory
# split video every 3 seconds

import gc
import os
import cv2
from typing import Generator

def split_video(video_path, output_dir: str=r"preprocess", interval=3):
    """
    Split a video into frames and save them to a directory with memory optimization.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the output frames.
        interval (int): Interval in seconds to split the video.

    Returns:
        str: Path to the directory where frames are saved.
    """
    try:
        # check if the video file exists
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file {video_path} not found.")

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        # get file name and create a subdirectory for the video
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        video_output_dir = os.path.join(output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Get the frame rate of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval)

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save the frame every 'frame_interval' frames
            # add timeframe to the filename
            if frame_count % frame_interval == 0:
                # Calculate the time in seconds
                time_in_seconds = frame_count / fps
                # Create a filename with the time in seconds 
                # use 4 digits for time_in_seconds and replace '.' with '_'
                # e.e, 1.5 -> 0001_5 , 2.2 -> 0002_2
                filename = os.path.join(video_output_dir, f"{video_name}_{int(time_in_seconds):0>10}.jpg")
                cv2.imwrite(filename, frame)
                # Explicitly delete frame after saving
                del frame
                gc.collect()

            frame_count += 1

    finally:
        # Ensure resources are released
        cap.release()
        gc.collect()

    print(f"Video split into frames and saved to {video_output_dir}")
    return video_output_dir


def load_prompts(prompt_path: str="video_surveillance_prompts.yml") -> str:
    """
    Load a prompt from a file.

    Args:
        prompt_path (str): Path to the prompt YAML file.        

    Returns:
        str: system_prompt, user_prompt.
    """
    # load yaml file
    import yaml

    with open(prompt_path, 'r') as file:
        prompts = yaml.safe_load(file)

    system_prompt = prompts.get('system_prompt', '')
    user_prompt = prompts.get('user_prompt', '')
    summary_sys_prompt = prompts.get('summary_sys_prompt', '')

    if not system_prompt or not user_prompt:
        raise ValueError("system_prompt and user_prompt must be provided in the YAML file.")
    
    return system_prompt, user_prompt, summary_sys_prompt