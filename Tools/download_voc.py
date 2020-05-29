"""
Script to download VOC data from torchvision library

"""

import torch
import torchvision
import subprocess


def main(args):
    VOCDataset_trainval = torchvision.datasets.VOCDetection(args.download_path, image_set='trainval', year='2007', download=True)
    VOCDataset_test = torchvision.datasets.VOCDetection(args.download_path, image_set='test', year='2007', download=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_path", help="Path to download VOC", default=".")

    args = parser.parse_args()
    main(args)