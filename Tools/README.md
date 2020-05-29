# Tools

## Walkthrough
In this directory multiple utility scripts are provided.

```convert_to_voc_annotations.py``` processed ExDark default annotations and converts them to VOC xml format.

```download_voc.py``` provides torchvision download script for VOC dataset used in this project.

```gan_dataset_creator_voc.py``` helps with creating the <em>target</em> dataset for GAN training from the PASCAL VOC dataset

```summary_statistics.py``` is used to create Table 1 in the report by summarizing the dataset.

```utils_fn.py``` contains a collection of utilities used to process datasets and transfer images.


All of these functions are implemented specifically to deal with the ExDark dataset, SSD and EnlightenGAN approaches.