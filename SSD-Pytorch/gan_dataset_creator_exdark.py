"""
Create source GAN Dataset by running SSD object detections over the ExDark dataset. This is a purely unsupervised approach and integrates
object context.

"""

from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from glob import glob
from random import shuffle
import shutil
import sys

sys.path.append('..')
sys.path.append('SSD-Pytorch')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

exdark_labels = ('bicycle', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',  'diningtable', 'dog', 'motorbike', 'person')


# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])




def detect(image_path, args, mode='train'):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """


    original_image = Image.open(image_path, mode='r')
    original_image = original_image.convert('RGB')

    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=args.min_score,
                                                             max_overlap=args.max_overlap, top_k=args.top_k)


    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.load_default()

    # Suppress specific classes, if needed
    # For each object detected, copy the image to directory corresponding to that object/label

    for i in range(det_boxes.size(0)):
        if args.suppress is not None:
            if det_labels[i] in args.suppress:
                continue

        if det_labels[i] in exdark_labels:
            shutil.copy(image_path, "{}/{}_{}/{}/{}".format(args.output_path, mode, args.tag, det_labels[i], image_path.split("/")[-1]))


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    
    parser.add_argument("--exdark_images_path", help="Path to Exdark images", default="/work/vq218944/MSAI/ExDarkVOC/JPEGImages")
    
    # A folder needs to be created where the output data is saved
    parser.add_argument("--output_path", help="Output data path", default="/work/vq218944/MSAI/EnlightenGAN_Data")
    # tag A meaning source data
    parser.add_argument("--tag", help="GAN dataset tag", default="A")
    # Train-test split for the GAN itself
    parser.add_argument("--split", help="Train-test split", default=0.8)
    # SSD model for detections
    parser.add_argument("--checkpoint", help="Model Checkpoint", default='/work/vq218944/MSAI/Models/checkpoint_ssd300.pth.tar')
    # Trainval file containing of list of images to use
    parser.add_argument("--trainvalfile", help="File with trainval list of images", default='/work/vq218944/MSAI/ExDarkVOC/ImageSets/Main/trainval.txt')
    
    # Detection parameters. A low score is set to get more instances.
    parser.add_argument("--min_score", help="Min score for detection", default=0.2, type=float)
    parser.add_argument("--max_overlap", help="Max overlap for NMS", default=0.5, type=float)
    parser.add_argument("--top_k", help="Max overlap for NMS", default=5, type=int)
    parser.add_argument("--suppress", help="Suppress any labels", default=None)


    args = parser.parse_args()
    # Load model checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    start_epoch = checkpoint['epoch'] + 1
    print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
    
    # Move model to device and load it in eval mode!
    model = checkpoint['model']
    model = model.to(device)
    model.eval()


    # Create subfolders for each label ( all the objects present in the dataset )
    for label in exdark_labels:
        if not(os.path.exists("{}/train_{}/{}".format(args.output_path, args.tag, label))):
            os.mkdir("{}/train_{}/{}".format(args.output_path, args.tag, label))

        if not(os.path.exists("{}/test_{}/{}".format(args.output_path, args.tag, label))):
            os.mkdir("{}/test_{}/{}".format(args.output_path, args.tag, label))


    # Get list of files
    files = glob("{}/*".format(args.exdark_images_path))

    with open(args.trainvalfile, "r") as fp:
        trainval_images = fp.read().splitlines() 

    # Filter files only present in train-val, test is untouched during GAN training
    files = [fn for fn in files if fn.split("/")[-1].split(".")[0] in trainval_images]

    # Shuffle files for GAN train-test which is ExDark train+val
    shuffle(files)
    train_files = files[:int(len(files)*args.split)]
    test_files = files[int(len(files)*args.split):]

    # Detect over the train and test files and copy images to folders corresponding to the objects detected. 
    print("\n Loading Train Detections")
    for fn in tqdm(train_files):
        detect(fn, args, mode='train')

    print("\n Loading Test Detections")
    for fn in test_files:
        detect(fn, args, mode='test')





