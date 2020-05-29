"""
Easily callable util functions file for different operations such as batch copy, create dataset from EnlightenGAN generated files etc.

To use this script cd into directory that contains this script then run,

python util_fns.py <name_of_fn_defined>

"""
import os 
from glob import glob
import shutil
from random import shuffle



def batch_copy():
    """
    Copy nested files from source_dir to dest_dir using os walk.
    Used to copy images/annotations from ExDark structure to flat VOC structure.
    """

    # Path of the ExDark dataset. This folder path should contain subfolders with different objects
    source_dir = "/work/vq218944/MSAI/ExDark"
    # Replace for annotations
    # source_dir = "/work/vq218944/MSAI/ExDark_Annno"

    # Path to an empty folder where the images/annotations are stored.
    dest_dir = "/work/vq218944/MSAI/ExDarkVOC/JPEGImages"
    # Replace for Annotations
    # dest_dir = "/work/vq218944/MSAI/ExDarkVOC/Annotations"

    for root, dirs, files in os.walk(source_dir):
        for fn in files:
            shutil.copy("{}/{}".format(root, fn), "{}/{}".format(dest_dir, fn.split("/")[-1]))


def copy_files_from_voc_list():
    """
    Copy files from source_dir to dest_dir if they are present in the voc_list. 
    To use this, create a folder with test_A, test_B subdirectories and provide the 
    location of test_A in dest_dir. Once the script is run, copy one image from test_A to test_B.
    The main folder is now the path to the test_dataset for evaluation.
    """
    voc_list = "/work/vq218944/MSAI/ExDarkVOC/ImageSets/Main/test.txt"
    source_dir = "/work/vq218944/MSAI/ExDarkVOC/JPEGImages"
    dest_dir = "/work/vq218944/MSAI/test_dataset/test_A"

    with open(voc_list, "r") as fp:
        split_images = fp.read().splitlines() 


    for fn in glob("{}/*".format(source_dir)):
        if fn.split("/")[-1].split(".")[0] in split_images:
            shutil.copy(fn, "{}/{}".format(dest_dir, fn.split("/")[-1]))
        
def copy_enlighten_generated_files():
    """
    Copy images from ablation folder in the EnlightenGAN to a copy of the ExDark VOC. 
    Benchmarks can then be run on this copy. 
    Provide the full path to folder containing images within the ablation folder as source_dir
    The JPEGImages folder within the VOC copy is the dest_dir
    """
    source_dir = "/home/vq218944/MSAI/Low-Light-Enhancement/EnlightenGAN/ablation/final_style/test_latest/images"
    dest_dir = "/work/vq218944/MSAI/ExDarkVOC_Final_Style/JPEGImages"

    for fn in glob("{}/*".format(source_dir)):
        if "fake" in fn:
            print("Copying file: {}".format(fn))
            new_fn = fn.split("/")[-1].strip("_fake_B.png")
            shutil.copy(fn, "{}/{}.png".format(dest_dir, new_fn))


def create_file_list():
    """
    Parse .txt file with trainval-test split information for different files.
    Save the trainval.txt and test.txt list of files in the destdir. This directory needs to exist.
     

    """
    dest_dir = "/work/vq218944/MSAI/ExDarkVOC_Enhanced/ImageSets/Main"

    image_class_list = "/work/vq218944/MSAI/imageclasslist.txt"

    with open(image_class_list, "r") as fp:
        image_list = fp.read().splitlines()

    trainval_files = []
    test_files = []
    
    for entry in image_list:
        image_info = entry.split(" ")
        filename = image_info[0].split(".")[0]
        split = int(image_info[-1])
        if split == 2 or split == 1:
            trainval_files.append(filename)

        if split == 3:
            test_files.append(filename)

    trainvaltxt = ""
    testtxt = ""
    for fn in trainval_files:
        trainvaltxt += "{}\n".format(fn)

    for fn in test_files:
        testtxt += "{}\n".format(fn)


    with open("{}/trainval.txt".format(dest_dir), "w") as fp:
        fp.write(trainvaltxt)

    with open("{}/test.txt".format(dest_dir), "w") as fp:
        fp.write(testtxt)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("util_fn", help="Util fn to be called from set of processing utilities")

    args = parser.parse_args()
    util_fn = eval(args.util_fn)
    util_fn()
