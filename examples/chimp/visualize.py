import os
import argparse
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime


def draw_landmarks_canonical(img_path, data, trans, output_dir):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.title(trans, fontsize=30)
    plt.imshow(image)

    for i_img in range(data.shape[0]):
        plt.scatter(data[i_img][0], data[i_img][1])

    name = datetime.now().isoformat(timespec='microseconds') 
    plt.savefig(output_dir + "/imgs_aggr/" + name + '.png', bbox_inches='tight')

def main(args):
    file_list_transform = os.path.join(args.input_dir, 'list_transform' + '.pickle')
    file_list_image = os.path.join(args.input_dir, 'list_image' + '.pickle')

    Path(args.input_dir + "/imgs_aggr").mkdir(parents=True, exist_ok=True)

    with open(file_list_transform, 'rb') as handle:
        list_transform = pickle.load(handle)

    with open(file_list_image, 'rb') as handle:
        list_image = pickle.load(handle)

    av_image = {}
    for img in list_image:
        for trans in list_image[img]:
            draw_landmarks_canonical(img, list_image[img][trans], trans, args.input_dir)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir", help="Input folder", dest="input_dir", type=str)

    args = parser.parse_args()
    main(args)