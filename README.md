# Quantization of pytorch-superpoint with LiteML

This is a fork of [pytorch-superpoint](https://github.com/eric-yyjau/pytorch-superpoint/tree/master) repo that demonstrates the usage of LiteML to perform PTQ and QAT on superpoint model.
The original repo is a PyTorch implementation of  "SuperPoint: Self-Supervised Interest Point Detection and Description." Daniel DeTone, Tomasz Malisiewicz, Andrew Rabinovich. [ArXiv 2018](https://arxiv.org/abs/1712.07629).


## Installation
### Requirements
- python == 3.6
- pytorch >= 1.1 (tested in 1.10.0)
- torchvision >= 0.3.0 (tested in 0.11.1)
- cuda (tested in cuda11.3)

Copy the provided LiteML whl file to this folder and then install using the commands below:
```
conda create --name py36-sp python=3.6
conda activate py36-sp
pip install -r requirements.txt
pip install liteml_sp-25.0.0-cp36-cp36m-linux_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu113
```

### Path setting
- paths for datasets ($DATA_PATH), logs are set in `setting.py`

### Dataset
Datasets should be downloaded into $DATA_PATH. The Synthetic Shapes dataset will also be generated there. The folder structure should look like:

```
datasets/ ($DATA_PATH)
|-- COCO
|   |-- train2014
|   |   |-- file1.jpg
|   |   `-- ...
|   |-- val2014
|       |-- file1.jpg
|       `-- ...
|-- HPatches
|   |-- i_ajuntament
|   `-- ...
|-- synthetic_shapes  # will be automatically created
```
- MS-COCO 2014 
    - [MS-COCO 2014 link](http://cocodataset.org/#download)
- HPatches
    - [HPatches link](http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz)



## Run the code
This repo supports:

1) Running PTQ on a pretrained float model and evaluating the model.

2) Loading a pretrained QAT model and evaluating the model.

3) Performing QAT on a pretrained float model using COCO dataset.

### 1) Running PTQ on a pretrained float model and evaluating the model
This step loads a pretrained superpoint model, wraps it with LiteML to perform PTQ with W8A8 configuration, exports the detections on HPatches dataset and finally evaluates the repeatbility.
#### Perform PTQ and export keypoints and descriptors
- download HPatches dataset (link above). Put in the $DATA_PATH. The general command is
```python export.py <export task> <config file> <export folder>```
```
python export.py export_descriptor configs/liteml_magicpoint_repeatability_heatmap_W8A8_PTQ.yaml W8A8_per_tensor_PTQ
```
#### Evaluate
The general command is
```python evaluation.py <path to npz files> [-r, --repeatibility | -o, --outputImg | -homo, --homography ]```
- Evaluate homography estimation/ repeatability/ matching scores ...
```
python evaluation.py logs/W8A8_per_tensor_PTQ/predictions --repeatibility
```

### 2) Loading a pretrained QAT model and evaluating the model
This step loads an already retrained model with QAT for W4A8 configuration. It then exports the detections on HPatches dataset and evaluates the repeatbility. The steps below are similar to step (1) but with different config file.
#### Load a pretrained QAT model and export keypoints and descriptors
- download HPatches dataset (link above) if haven't done in step (1). Put in the $DATA_PATH. The general command is
```python export.py <export task> <config file> <export folder>```
```
python export.py export_descriptor configs/liteml_magicpoint_repeatability_heatmap_W4A8_QAT.yaml W4A8_per_channel_QAT_170800
```
#### Evaluate
```
python evaluation.py logs/W4A8_per_channel_QAT_170800/predictions --repeatibility
```

### 3) Performing QAT on a pretrained float model using COCO dataset
#### a) Exporting pseudo ground truth labels on MS-COCO
This is the step of homography adaptation(HA) to export pseudo ground truth for joint training. This is done on a float model before the QAT stage.
- make sure the pretrained model in config file is correct
- make sure COCO dataset is in '$DATA_PATH' (defined in setting.py)
<!-- - you can export hpatches or coco dataset by editing the 'task' in config file -->
- config file:
```
export_folder: <'train' | 'val'>  # set export for training or validation
```
##### General command:
```
python export.py <export task>  <config file>  <export folder> [--outputImg | output images for visualization (space inefficient)]
```
##### export coco - do on training set
- Edit 'export_folder' to 'train' in 'magicpoint_coco_export.yaml'
```
python export.py export_detector_homoAdapt configs/magicpoint_coco_export.yaml magicpoint_synth_homoAdapt_coco
```
##### export coco - do on validation set 
- Edit 'export_folder' to 'val' in 'magicpoint_coco_export.yaml'
```
python export.py export_detector_homoAdapt configs/magicpoint_coco_export.yaml magicpoint_synth_homoAdapt_coco
```


#### b) Performing QAT using the exported pseudo ground truth labels on MS-COCO
You need pseudo ground truth labels to traing detectors. Labels can be exported from step a) Then, you need to set config file before training.
- config file
  - root: specify your labels root
  - root_split_txt: where you put the train.txt/ val.txt split files (no need for COCO, needed for KITTI)
  - labels: the exported labels from homography adaptation
  - pretrained: specify the pretrained float model (you can train from scratch)
- 'eval': turn on the evaluation during training 

#### General command
```
python train4.py <train task> <config file> <export folder> --eval
```

#### COCO
```
python train4.py train_joint configs/liteml_superpoint_coco_train_heatmap.yaml superpoint_coco_liteml_w4a8_qat --eval --debug
```

## Pretrained models
### Current best models
- *COCO dataset - float model*
```logs/superpoint_coco_heat2_0/checkpoints/superPointNet_170000_checkpoint.pth.tar```

- *COCO dataset - W4A8 QAT model, retrained for 800 iterations on top of the float model above*
```logs/superpoint_coco_liteml_w4a8_qat/checkpoints/superPointNet_170800_checkpoint.pth.tar```

### model from magicleap
```pretrained/superpoint_v1.pth```


## Citations
Please cite the original paper.
```
@inproceedings{detone2018superpoint,
  title={Superpoint: Self-supervised interest point detection and description},
  author={DeTone, Daniel and Malisiewicz, Tomasz and Rabinovich, Andrew},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  pages={224--236},
  year={2018}
}
```


# Credits
This is a fork of the [pytorch-superpoint](https://github.com/eric-yyjau/pytorch-superpoint/tree/master) developed by [You-Yi Jau](https://github.com/eric-yyjau) and [Rui Zhu](https://github.com/Jerrypiglet). Please contact You-Yi for any problems. 
Again the work is based on Tensorflow implementation by [Rémi Pautrat](https://github.com/rpautrat) and [Paul-Edouard Sarlin](https://github.com/Skydes) and official [SuperPointPretrainedNetwork](https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork).
