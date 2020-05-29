"""
Script to run CLAHE over a folder and save it in EnlightenGAN ablation format! 
Easy to compare and process with EnlightenGAN outputs.

"""

import numpy as np
import cv2
from glob import glob
import os

def main(args):
    for fn in glob("{}/*".format(args.data_folder)):
        print("Reading file: ", fn)
        img = cv2.imread(fn)

        # create a CLAHE object (Arguments are optional).
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
        img_lab[:,:,0] = clahe.apply(img_lab[:,:,0])
        img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
        
        os.makedirs(args.output_folder, exist_ok=True)
        cv2.imwrite('{}/{}_fake_B.png'.format(args.output_folder, fn.split("/")[-1].split(".")[0]),img)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # Data folder provided in the similar test_A format
    parser.add_argument("--data_folder", help="Folder to data to run CLAHE on", default="/home/vq218944/Downloads/DICM/test_A")
    # Output folder to save the processed images. It is saved under ablation with similar format as the enlightenGAN generated enhancements.
    # This is done to allow easy integrating into the subsequent benchmarking process.
    parser.add_argument("--output_folder", help="Folder as the output", default="/home/vq218944/MSAI/Low-Light-Enhancement/EnlightenGAN/ablation/clahe/images")
    args = parser.parse_args()
    main(args)
