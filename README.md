# StarFlow

## Environment Setup

Note: the code in this repo has been tested on Ubuntu 18.04 with  Python 3.8, CUDA 11.1/11.2 and PyTorch 1.9.0. It  may work for other setups, but has not been tested. 

## Prerequisities

Our model is trained and tested under:

- Python 3.8.12
- NVIDIA GPU + CUDA CuDNN
- PyTorch 1.9.0
- scipy
- tqdm
- sklearn
- numba
- cffi
- pypng
- pptk
- thop
- open3d for visualization

 Please follow this [repo](https://github.com/sshaoshuai/Pointnet2.PyTorch) or the instructions below for compiling the furthest point sampling, grouping and gathering operation for PyTorch.

```
cd pointnet2
python setup.py install
cd ../
```



## Data preprocess

### FlyThings3Ds and StereoKITTIs

We make FT3Ds and KITTIs datasets by leveraging the equivalent preprocessing steps in [HPLFlowNet](https://web.cs.ucdavis.edu/~yjlee/projects/cvpr2019-HPLFlowNet.pdf) and [PointPWCNet](https://github.com/DylanWusee/PointPWC).

Here we copy the FT3Ds and KITTIs preprocessing instructions here for your convinience from [HPLFlowNet](https://github.com/laoreja/HPLFlowNet).

- FlyingThings3D:
  Download and unzip the "Disparity", "Disparity Occlusions", "Disparity change", "Optical flow", "Flow Occlusions" for DispNet/FlowNet2.0 dataset subsets from the [FlyingThings3D website](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) (we used the paths from [this file](https://lmb.informatik.uni-freiburg.de/data/FlyingThings3D_subset/FlyingThings3D_subset_all_download_paths.txt), now they added torrent downloads)
  . They will be upzipped into the same directory, `RAW_DATA_PATH`. Then run the following script for 3D reconstruction:

```bash
python3 data_preprocess/process_flyingthings3d_subset.py --raw_data_path RAW_DATA_PATH --save_path SAVE_PATH/FlyingThings3D_subset_processed_35m --only_save_near_pts
```

- KITTI Scene Flow 2015
  Download and unzip [KITTI Scene Flow Evaluation 2015](http://www.cvlibs.net/download.php?file=data_scene_flow.zip) to directory `RAW_DATA_PATH`.
  Run the following script for 3D reconstruction:

```bash
python3 data_preprocess/process_kitti.py RAW_DATA_PATH SAVE_PATH/KITTI_processed_occ_final
```

### FlyThings3Do and StereoKITTIo

We follow [FlowNet3D](https://github.com/xingyul/flownet3d) to process FlyThings and KITTI respectively to make datasets FT3Do and KITTIo.

### SF KITTI 

The SFKITTI dataset is proposed in paper [FH-Net](https://link.springer.com/chapter/10.1007/978-3-031-19842-7_13). And we apply [original methods](https://github.com/pigtigger/FH-Net) to obtain SFKITTI dataset.

### LiDAR KITTI

For fair comparisions with previous methods, we follow [Rigid3DSF](https://github.com/zgojcic/Rigid3DSceneFlow) to fatch the LiDAR KITTI dataset.

The **details** of datasets processing are in directory of our code.

```
./datasets/flyingthings3d_subset.py
```



## Evaluate

First set the `data_root` in the configuration file to `SAVE_PATH` in the data preprocess section before evaluation. Second choose the pretrained model from scatch. And we provide one pretrained model in our directory ```./pretrain_weights/```. Finally please run the following instrcutions for evaluating.

```
python3 evaluate_starflow.py config_evaluate_starflow.yaml
```



## Train

If you need a newly trained model from scatch, please set `data_root` in the configuration file to `SAVE_PATH` in the data preprocess section before evaluation at the first. Note that to pay attention to selecting the dataset that needs training. And then excute following instructions.

```bash
python3 train_starflow.py config_train_starflow.yaml
```

# STARFlow++: with a refine module to reduce EPE3D.

If you want a two-stage scene flow network to further improve performance, we provide a refine module to refine the scene flow in the second stage. Please follow the command line below for training and evaluation.

```
python3 train_starflow_refine.py config_train_starflow_refine.yaml
```

```
python3 evaluate_starflow_refine.py  config_evaluate_starflow.yaml
```



## Acknowledgement

In this project we use parts of the official implementations of: 

- [PointPWC](https://github.com/DylanWusee/PointPWC)
- [Rigid3DSceneFlow](https://github.com/zgojcic/Rigid3DSceneFlow)
- [FlowNet3D](https://github.com/xingyul/flownet3d)
- [FH-Net](https://github.com/pigtigger/FH-Net)

We thank the respective authors for open sourcing their methods. 



