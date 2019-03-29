# Large-Scale-VRD.pytorch

This is a PyTorch implementation for [Large-scale Visual Relationship Understanding, AAAI2019](https://arxiv.org/abs/1804.10660).

This code is for the VG200 and VRD datasets only. For results on VG80K please refer to the [Caffe2 implemntation](https://github.com/facebookresearch/Large-Scale-VRD).

We referred to [Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch) and borrowed their framework for this project, so there are a lot overlaps between these two.

## Requirements
* Python 3
* Python packages
  * pytorch 0.4.0 or 0.4.1.post2 (not guaranteed to work on newer versions)
  * cython
  * matplotlib
  * numpy
  * scipy
  * opencv
  * pyyaml
  * packaging
  * [pycocotools](https://github.com/cocodataset/cocoapi)
  * tensorboardX
  * tqdm
  * pillow
  * scikit-image
  * gensim
* An NVIDIA GPU and CUDA 8.0 or higher. Some operations only have gpu implementation.

## Compilation
Compile the CUDA code in the Detectron submodule and in the repo:
```
cd $ROOT/lib
sh make.sh
```

## Annotations

Create a data folder at the top-level directory of the repository:
```
# ROOT=path/to/cloned/repository
cd $ROOT
mkdir data
```

### Visual Genome
Download it [here](https://drive.google.com/open?id=1VDuba95vIPVhg5DiriPtwuVA6mleYGad). Unzip it under the data folder. You should see a `vg` folder unzipped there. It contains .json annotations that suit the dataloader used in this repo.

### Visual Relation Detection
Download it [here](https://drive.google.com/open?id=1BUZIVOCEp_-_e9Rs4hVgmbKjLhR2aUT6). Unzip it under the data folder. You should see a `vrd` folder unzipped there. It contains .json annotations that suit the dataloader used in this repo.

## Images

### Visual Genome
Create a folder for all images:
```
# ROOT=path/to/cloned/repository
cd $ROOT/data/vg
mkdir VG_100K
```
Download Visual Genome images from the [official page](https://visualgenome.org/api/v0/api_home.html). Unzip all images (part 1 and part 2) into `VG_100K/`. There should be a total of 108249 files.

### Visual Relation Detection
Download Visual Relation Detection images from the [here](https://drive.google.com/open?id=1M015ElsLR6SAuCD_bhsnn5Pi5T4WQ8p6). Unzip it under the `vrd` folder and you should see `train_images` and `val_images` there. Inside them are images with cleaned file names (the original VRD images use hashes as names and we convert them to numbers).

## Pre-trained Object Detection Models
Download pre-trained object detection models [here](https://drive.google.com/open?id=16JVQkkKGfiGt7AUt789pUPX3o84Cl2hL). Unzip it under the root directory and you should see a `detection_models` folder there.

## Our Trained Relationship Detection Models
Download our trained models [here](https://drive.google.com/open?id=1bYq02eInCqT4D1Brnv7y18_Uosw23LqZ). Unzip it under the root folder and you should see a `trained_models` folder there.

## Directory Structure
The final directories for data and detection models should look like:
```
|-- detection_models
|   |-- vg
|   |   |-- VGG16
|   |   |   |-- model_step479999.pth
|   |   |-- X-101-64x4d-FPN
|   |   |   |-- model_step119999.pth
|   |-- vrd
|   |   |-- VGG16
|   |   |   |-- model_step4499.pth
|-- data
|   |-- vg
|   |   |-- VG_100K    <-- (contains Visual Genome all images)
|   |   |-- rel_annotations_train.json
|   |   |-- rel_annotations_val.json
|   |   |-- ...
|   |-- vrd
|   |   |-- train_images    <-- (contains Visual Relation Detection training images)
|   |   |-- val_images    <-- (contains Visual Relation Detection validation images)
|   |   |-- new_annotations_train.json
|   |   |-- new_annotations_val.json
|   |   |-- ...
|-- trained_models
|   |-- e2e_relcnn_VGG16_8_epochs_vg_y_loss_only
|   |   |-- model_step125445.pth
|   |-- e2e_relcnn_X-101-64x4d-FPN_8_epochs_vg_y_loss_only
|   |   |-- model_step125445.pth
|   |-- e2e_relcnn_VGG16_8_epochs_vrd_y_loss_only
|   |   |-- model_step7559.pth
|   |-- e2e_relcnn_VGG16_8_epochs_vrd_y_loss_only_w_freq_bias
|   |   |-- model_step7559.pth
```

## Evaluating Pre-trained Relationship Detection models

DO NOT CHANGE anything in the provided config files(configs/xx/xxxx.yaml) even if you want to test with less or more than 8 GPUs. Use the environment variable `CUDA_VISIBLE_DEVICES` to control how many and which GPUs to use. Remove the
`--multi-gpu-test` for single-gpu inference.

### Visual Genome
**NOTE:** May require at least 64GB RAM to evaluate on the Visual Genome test set

We use three evaluation metrics for Visual Genome:
1. SGDET: predict all the three labels and two boxes
1. SGCLS: predict subject, object and predicate labels given ground truth subject and object boxes
1. PRDCLS: predict predicate labels given ground truth subject and object boxes and labels

To test a trained model using a VGG16 backbone with "SGDET", run
```
python ./tools/test_net_rel.py --dataset vg --cfg configs/vg/e2e_relcnn_VGG16_8_epochs_vg_y_loss_only.yaml --load_ckpt trained_models/e2e_relcnn_VGG16_8_epochs_vg_y_loss_only/model_step125445.pth --output_dir Outputs/e2e_relcnn_VGG16_8_epochs_vg_y_loss_only --multi-gpu-testing --do_val
```
Use `--use_gt_boxes` option to test it with "SGCLS"; use `--use_gt_boxes --use_gt_labels` options to test it with "PRDCLS".

To test a trained model using a vg_X-101-64x4d-FPN backbone with "SGDET", run
```
python ./tools/test_net_rel.py --dataset vg --cfg configs/vg/e2e_relcnn_X-101-64x4d-FPN_8_epochs_vg_y_loss_only.yaml --load_ckpt trained_models/vg_X-101-64x4d-FPN/model_step125445.pth --output_dir Outputs/e2e_relcnn_X-101-64x4d-FPN_8_epochs_vg_y_loss_only --multi-gpu-testing --do_val
```
Use `--use_gt_boxes` option to test it with "SGCLS"; use `--use_gt_boxes --use_gt_labels` options to test it with "PRDCLS".

### Visual Relation Detection
To test a trained model using a VGG16 backbone, run
```
python ./tools/test_net_rel.py --dataset vrd --cfg configs/vrd/e2e_relcnn_VGG16_8_epochs_vrd_y_loss_only.yaml --load_ckpt trained_models/e2e_relcnn_VGG16_8_epochs_vrd_y_loss_only/model_step7559.pth --output_dir Outputs/e2e_relcnn_VGG16_8_epochs_vrd_y_loss_only --multi-gpu-testing --do_val
```

## (Optional) Training Object Detection Models
This repo provides code for training object detectors for VG using a ResNeXt-101-64x4d-FPN backbone.

First download weights of a ResNeXt-101-64x4d-FPN trained on ImageNet [here](https://drive.google.com/open?id=1HvznYV86YJp6wfNj7ksFw1okvRz8ZuwN). Unzip it under the `data` directory and you should see a `detectron_model` folder.

To train the object detector, run
```
python ./tools/train_net_step.py --dataset vg --cfg configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_vg.yaml --nw 8 --use_tfboard
```

## Training Relationship Detection Models

The section provides the command-line arguments to train our relationship detection models given the pre-trained object detection models described above. **Note:** We do not train object detectors here. We only use trained object detectors (provided in `detection_models/`) to initialize our to-be-trained relationship models.

DO NOT CHANGE anything in the provided config files(configs/xx/xxxx.yaml) even if you want to train with less or more than 8 GPUs. Use the environment variable `CUDA_VISIBLE_DEVICES` to control how many and which GPUs to use.

With the following command lines, the training results (models and logs) should be in `$ROOT/Outputs/xxx/` where `xxx` is the .yaml file name used in the command without the ".yaml" extension. If you want to test with your trained models, simply run the test commands described above by setting `--load_ckpt` as the path of your trained models.

### Visual Genome
To train our relationship network using a VGG16 backbone, run
```
python tools/train_net_step_rel.py --dataset vg --cfg configs/vg/e2e_relcnn_VGG16_8_epochs_vg_y_loss_only.yaml --nw 8 --use_tfboard
```

To train our relationship network using a ResNeXt-101-64x4d-FPN backbone, run
```
python tools/train_net_step_rel.py --dataset vg --cfg configs/vg/e2e_relcnn_X-101-64x4d-FPN_8_epochs_vg_y_loss_only.yaml --nw 8 --use_tfboard
```

### Visual Relation Detection
To train our relationship network using a VGG16 backbone, run
```
python tools/train_net_step_rel.py --dataset vrd --cfg configs/vrd/e2e_relcnn_VGG16_8_epochs_vrd_y_loss_only.yaml --nw 8 --use_tfboard
```

## Acknowledgements
This repository uses code based on the [Neural-Motifs](https://github.com/rowanz/neural-motifs) source code from Rowan Zellers, as well as
code from the [Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch) repository by Roy Tseng.

## Citing
If you use this code in your research, please use the following BibTeX entry.
```
@conference{zhang2018large,
  title={Large-Scale Visual Relationship Understanding},
  author={Zhang, Ji and Kalantidis, Yannis and Rohrbach, Marcus and Paluri, Manohar and Elgammal, Ahmed and Elhoseiny, Mohamed},
  booktitle={AAAI},
  year={2019}
}
