import pandas as pd
import numpy as np
import os
import glob
import skvideo.io
from pathlib import Path
from tqdm import tqdm
import sys
import argparse
import mlflow

parser = argparse.ArgumentParser()

parser.add_argument("--step", default = "train" , help="train or inference step")
parser.add_argument("--ann_path", default = "./Release_v1/annotations/bounding_box_gt" , help="annotation path")
parser.add_argument("--video_path", default = "./Release_v1/videos/fps1/" , help="video path")
parser.add_argument("--frame_path", default = "./Release_v1/frames/", help="output dir to store frames")

args = parser.parse_args(sys.argv[1:])

# Different labels
uniq_labels = ["dropped", "grabbed_-_left", "grabbed_-_right", "grabbed_-_both"]
# Creating hashmaps to mapp label strings to indices and vide versa
labels2idx = {uniq_labels[i]: i for i in range(len(uniq_labels))}
idx2labels = {v: k for k, v in labels2idx.items()}

def videoToframeTrain(videoFrame, videoJson, frame_path, labels2idx):
        

    frames = skvideo.io.vread(videoFrame) # to read the video data
    base_name = Path(videoFrame).stem     # the basename of the file
    df = pd.read_json(videoJson).T        # to transpose the DataFrame
    
    # print("frames shape: ", frames.shape)

    max_labeled_frames = df["frame_id"].max() # finding the maximum numder of frames being labeled
    # print("max_labeled_frames: ", max_labeled_frames)
    

    for i in range(0, int(max_labeled_frames)):
        frame = frames[i,...]
        
        label = df.loc[(df["frame_id"] == i+1) & (df["obj_class"] == "needle"), "orientation"]
        if not label.empty:
            label = label.item()

        
            filename = os.path.join(frame_path, base_name + f"_frame_id={i}")
            np.savez(filename, img = frame, label = labels2idx[label])


def videoToframeTest(videoFrame, frame_path):
        

    frames = skvideo.io.vread(videoFrame) # to read the video data
    base_name = Path(videoFrame).stem     # the basename of the file
    
    # print("frames shape: ", frames.shape)

    for i in range(0, int(frames.shape[0])):
        frame = frames[i,...]
        
        filename = os.path.join(frame_path, base_name + f"_frame_id={i}")
        np.savez(filename, img = frame, label = -1)  # setting the test label to be negative one


if __name__ == "__main__":
    
    np.random.seed(40)

    model_step = args.step

    frame_path = args.frame_path
    ann_path = args.ann_path
    video_path = args.video_path

    
    if not os.path.exists(frame_path):
        # Create a new directory because it does not exist 
        os.makedirs(frame_path)
        print(f"The new directory, '{frame_path}'is created!")

    annList = glob.glob(ann_path + "/*.json")
    videoList = glob.glob(video_path + "/*.mp4")
    annList.sort()
    videoList.sort()

    if model_step == "train":

        for i in tqdm(range(len(videoList))): 
            videoToframeTrain(videoList[i], annList[i], frame_path, labels2idx)
    
    elif model_step == "test":
        for i in tqdm(range(len(videoList))): 
            videoToframeTest(videoList[i], frame_path)