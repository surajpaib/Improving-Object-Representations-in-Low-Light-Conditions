"""
Script to process ExDark dataset as released on https://github.com/cs-chan/Exclusively-Dark-Image-Dataset to PASCAL VOC Format

11/12 classes exist in VOC and only those classes are selected here. 
"""
import os
from xml.etree.ElementTree import Element, SubElement, Comment, tostring, ElementTree
from glob import glob
from PIL import Image 


# VOC Labels overlapping with labels in the ExDark dataset. Objects are renamed based on their exact names in PASCAL VOC
voc_labels = {"Bicycle": 'bicycle', "Boat": 'boat', "Bottle": 'bottle', "Bus": 'bus', "Car": 'car', "Cat": 'cat', "Chair": 'chair',  "Table": 'diningtable', 
            "Dog":'dog', "Motorbike": 'motorbike', "People": 'person'}


def main(args):

    # Loop over txt annotations in the folder
    for idx, fn in enumerate(glob("{}/*".format(args.exdark_annotations_folder))):

        # Create xml tree based on the VOC XML formatting
        annotation = Element('annotation')
        tree = ElementTree(annotation)

        folder = SubElement(annotation, 'folder')
        folder.text = "ExDarkVOC"

        source = SubElement(annotation, 'source')
        source.text = "Exclusively Dark Image Dataset: https://github.com/cs-chan/Exclusively-Dark-Image-Dataset"

        owner = SubElement(annotation, 'owner')
        owner.text = "Loh, Yuen Peng and Chan, Chee Seng"

        # Image is read to get the exact size
        size = SubElement(annotation, 'size')
        im = Image.open(glob("{}/JPEGImages/{}*".format(args.exdark_annotations_folder.rsplit("/", 1)[0], fn.split("/")[-1].split(".")[0]))[0])  


        filename = SubElement(annotation, 'filename')
        filename.text = glob("{}/JPEGImages/{}*".format(args.exdark_annotations_folder.rsplit("/", 1)[0], fn.split("/")[-1].split(".")[0]))[0]
        

        width = SubElement(size, 'width')
        width.text = str(im.size[0])
        height = SubElement(size, 'height')
        height.text = str(im.size[0])

        depth = SubElement(size, 'depth')
        depth.text = str(3)

        owner = SubElement(annotation, 'segmented')
        owner.text = str(0)

        # The annotation files are opened and objects in the file are mapped with VOC names and added to the xml tree
        with open(fn, "r") as fp:
            for line in fp.readlines()[1:]:
                ann_object = line.split()
                name = ann_object[0]
                if name in voc_labels:
                    annotated_object = SubElement(annotation, 'object')

                    object_name = voc_labels[name]
                    
                    obj_name = SubElement(annotated_object, 'name')
                    obj_name.text = object_name

                    # ExDark bbox convention changed to VOC convention
                    xmin = str(int(ann_object[1]))
                    xmax = str(int(ann_object[1]) + int(ann_object[3]))
                    ymin = str(int(ann_object[2]))
                    ymax = str(int(ann_object[2]) + int(ann_object[4]))

                    difficult = SubElement(annotated_object, 'difficult')
                    difficult.text = str(0)

                    bndbox = SubElement(annotated_object, 'bndbox')
                    xmin_el = SubElement(bndbox, 'xmin')
                    xmin_el.text = xmin
                    xmax_el = SubElement(bndbox, 'xmax')
                    xmax_el.text = xmax
                    ymin_el = SubElement(bndbox, 'ymin')
                    ymin_el.text = ymin
                    ymax_el = SubElement(bndbox, 'ymax')
                    ymax_el.text = ymax


        # Save xml trees to xml files
        with open("{}/{}.xml".format(args.exdark_annotations_folder,  fn.split("/")[-1].split(".")[0]), "wb") as fp:
            tree.write(fp)

        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Folder containing annotations in the ExDark format (but with a flat structure) 
    parser.add_argument("--exdark_annotations_folder", help="Path to Exdark dataset annotations", default="/work/vq218944/MSAI/ExDarkVOC/Annotations")
    
    # Output folder to save xml files needs to be provided.
    parser.add_argument("--output_path", help="Path to save output", default="/work/vq218944/MSAI/ExDarkVOC/Annotations")

    args = parser.parse_args()
    main(args)




