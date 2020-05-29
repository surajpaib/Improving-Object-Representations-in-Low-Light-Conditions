"""
Evaluation script to run SSD over ExDark dataset. Results are grouped by lighting conditions!

"""

from utils import *
from datasets import PascalVOCDataset
from tqdm import tqdm
from pprint import PrettyPrinter
import pandas as pd

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Parameters
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 64
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(args):
    """
    Evaluation scripts for exdark data. Refer to argparse arguments at teh end of the script
    """

    # Load model checkpoint that is to be evaluated

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = checkpoint['model']
    model = model.to(device)

    # Load test data

    test_dataset = PascalVOCDataset(args.data_folder,
                                    split='test',
                                    keep_difficult=keep_difficult)


    with open(args.exdark_metadata, "r") as fp:
        image_metadata = fp.read().splitlines()


    lighting_states = ['Low', 'Ambient', 'Object', 'Single', 'Weak', 'Strong', 'Screen', 'Window', 'Shadow', 'Twilight']
    lighting_indices = [[] for i in range(10)]
    indoor_outdoor = ['Indoor', 'Outdoor']
    location_indices = [[] for i in range(2)]

    for dataset_idx, _image in enumerate(test_dataset.images):
        for _metadata in image_metadata:
            metadata = _metadata.split(" ")
            if metadata[0].split(".")[0] == _image.split("/")[-1].split(".")[0]:
                lighting_indices[int(metadata[2]) - 1].append(dataset_idx)


    AP_dict = {}

    for state_idx, sets in enumerate(lighting_indices):


        print("\n \n Lighting State: ", lighting_states[state_idx])
        subset = torch.utils.data.Subset(test_dataset, sets)

        test_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False,
                                                collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)



        # Make sure it's in eval mode
        model.eval()

        # Lists to store detected and true boxes, labels, scores
        det_boxes = list()
        det_labels = list()
        det_scores = list()
        true_boxes = list()
        true_labels = list()
        true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py

        with torch.no_grad():
            # Batches
            for i, (images, boxes, labels, difficulties) in enumerate(test_loader):

                images = images.to(device)  # (N, 3, 300, 300)

                # Forward prop.
                predicted_locs, predicted_scores = model(images)

                # Detect objects in SSD output
                det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                            min_score=args.min_score, max_overlap=0.45,
                                                                                            top_k=200)
                # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

                # Store this batch's results for mAP calculation
                boxes = [b.to(device) for b in boxes]
                labels = [l.to(device) for l in labels]
                difficulties = [d.to(device) for d in difficulties]

                det_boxes.extend(det_boxes_batch)
                det_labels.extend(det_labels_batch)
                det_scores.extend(det_scores_batch)
                true_boxes.extend(boxes)
                true_labels.extend(labels)
                true_difficulties.extend(difficulties)
            
            
            # Calculate aP for exdark and add to dictionary with lighting condition.
            APs = calculate_mAP_exdark(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)
            APs["lighting"] = lighting_states[state_idx]
        
        # Print AP for each class
        pp.pprint(APs)

        if AP_dict:
            for key in AP_dict:
                AP_dict[key].append(APs[key])
        else:
            AP_dict = {}
            for key in APs:
                AP_dict[key] = [APs[key]]

    df = pd.DataFrame.from_dict(AP_dict)
    df.to_csv("{}_{}_results.csv".format(args.data_folder.split("/")[-1], args.min_score))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--min_score", help="Min score @ eval", default=0.01, type=float)
    
    # SSD checkpoint as downloaded in the setting up ssd section
    parser.add_argument("--checkpoint", help="Model checkpoint for eval", default="/work/vq218944/MSAI/Models/checkpoint_ssd300.pth.tar")
    
    # Path to the generated json files from the previous step
    parser.add_argument("--data_folder", help="Folder with train_objects and train_images json", default='/work/vq218944/MSAI/ssd_jsons/SSD_ExDark')
    
    # The imageclasslist.txt file downloaded for enlightenGAN trainval processing. This file also contains information about lighting conditions and class distributions
    parser.add_argument("--exdark_metadata", help="txt file with metadata for ExDark", default='/work/vq218944/MSAI/imageclasslist.txt')

    args = parser.parse_args()  

    evaluate(args)
