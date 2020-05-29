import os 
 
def summary(args):
    with open(args.exdark_metadata, "r") as fp:
        image_metadata = fp.read().splitlines()


    lighting_states = ['Low', 'Ambient', 'Object', 'Single', 'Weak', 'Strong', 'Screen', 'Window', 'Shadow', 'Twilight']
    train_summary_dict = {k:0 for k in lighting_states}
    test_summary_dict = {k:0 for k in lighting_states}

    indoor_outdoor = ['Indoor', 'Outdoor']
    location_indices = [[] for i in range(2)]

    for _metadata in image_metadata:
        metadata = _metadata.split(" ")

        if metadata[-1] == "3":
            test_summary_dict[lighting_states[int(metadata[2]) - 1]] += 1
        else:
            train_summary_dict[lighting_states[int(metadata[2]) - 1]] += 1

    print(train_summary_dict)
    print(test_summary_dict)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--exdark_metadata", help="Path to the Exdark metadata", default="/work/vq218944/MSAI/imageclasslist.txt")

    args = parser.parse_args()  

    summary(args)