"""
Create easily parsable data lists from PASCAL VOC annotation format
"""

from utils import create_data_lists

def main(args):

    create_data_lists(voc07_path=args.voc_path,
                      output_folder=args.output_path, mode=args.mode)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    # Keep as exdark
    parser.add_argument("--mode", help="Either VOC or ExDark", default="exdark")

    # Provide the path to the VOC dataset to convert to the json format
    parser.add_argument("--voc_path", help="Path to VOC dataset, trainval+test", default="/work/vq218944/MSAI/VOCdevkit/VOC2007")
    
    # The path to store these jsons. This needs to be provided to the next step of the evaluation process.
    parser.add_argument("--output_path", help="Path to save output", default="/work/vq218944/MSAI/")

    args = parser.parse_args()
    main(args)

