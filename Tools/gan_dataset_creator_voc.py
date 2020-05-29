"""
Utility to create GAN object paired dataset from labelled VOC data. Here this is the target. 
Pre-existing labelled data is chosen as target!

"""
import os
import xml.etree.ElementTree as ET
from glob import glob
from random import shuffle
import shutil

voc_labels = ('bicycle', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',  'diningtable', 'dog', 'motorbike', 'person')
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0


def split_folders(args):
    """
    Read PASCAL VOC format data from XML and generate folders with images for each specific object detected

    """
    images_path = "{}/JPEGImages".format(args.annotation_path.rsplit("/", 1)[0])

    # Create train and test directories for all labels (objects in the dataset)
    for label in voc_labels:
        if not(os.path.exists("{}/train_{}/{}".format(args.output_path, args.tag, label))):
            os.mkdir("{}/train_{}/{}".format(args.output_path, args.tag, label))

        if not(os.path.exists("{}/test_{}/{}".format(args.output_path, args.tag, label))):
            os.mkdir("{}/test_{}/{}".format(args.output_path, args.tag, label))

    # Shuffle files to create train-test split for GAN training
    files = glob("{}/*".format(args.annotation_path))
    shuffle(files)
    train_files = files[:int(len(files)*args.split)]
    test_files = files[int(len(files)*args.split):]


    # Parse XML tree in the annotations to identify what objects are present.
    # For each object present, copy the image to directory corresponding to that object/label
    for fn in train_files:
        tree = ET.parse(fn)
        root = tree.getroot()


        filename = fn.split("/")[-1].split(".")[0] + '.jpg'
        print(filename)

        for object in root.iter('object'):


            label = object.find('name').text.lower().strip()
            if label not in label_map:
                continue

            shutil.copy("{}/{}".format(images_path, filename), "{}/train_{}/{}/{}".format(args.output_path, args.tag, label, filename))


    for fn in test_files:
        tree = ET.parse(fn)
        root = tree.getroot()


        filename = fn.split("/")[-1].split(".")[0] + '.jpg'
        print(filename)

        for object in root.iter('object'):


            label = object.find('name').text.lower().strip()
            if label not in label_map:
                continue

            shutil.copy("{}/{}".format(images_path, filename), "{}/test_{}/{}/{}".format(args.output_path, args.tag, label, filename))



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()


    # Path to the VOC annotations and output folder to place the object-context grouped images need to be provided.
    parser.add_argument("--annotation_path", help="Path to the VOC Annotations", default="/work/vq218944/MSAI/VOCdevkit/VOC2007/Annotations")
    parser.add_argument("--output_path", help="Output data path", default="/work/vq218944/MSAI/EnlightenGAN_Data")
    parser.add_argument("--tag", help="GAN dataset tag", default="B")
    parser.add_argument("--split", help="Train-test split", default=0.0)

    args = parser.parse_args()  

    split_folders(args)