# OPGP

This repo is the implementation of:

**Occupancy Prediction-Guided Neural Planner for Autonomous Driving**
<br> [Haochen Liu](https://scholar.google.com/citations?user=iizqKUsAAAAJ&hl=en), [Zhiyu Huang](https://mczhi.github.io/),  [Chen Lv](https://scholar.google.com/citations?user=UKVs2CEAAAAJ&hl=en) 
<br> [AutoMan Research Lab, Nanyang Technological University](https://lvchen.wixsite.com/automan)
<br> **[[arXiv]](https://arxiv.org/abs/2305.03303)**&nbsp; **[[Zhihu]](https://zhuanlan.zhihu.com/p/630045890)**&nbsp;

- Code is now released 😀!

## Overview
In this repository, you can expect to find the following features 🤩:
* Pipelines for data process and training
* Open-loop evaluations
  
Not included 😵:
* Model weights (Due to license from WOMD)
* Real-time planning (Codes are not optimized for real-time performance)

## Experiment Pipelines

### Dataset and Environment


- Downloading [Waymo Open Motion Dataset](https://waymo.com/open/download/) v1.1. Utilize data from ```scenario/training_20s``` for train set, and data from ```scenario/validation``` for val & test.

- Clone this repository and install required packages.

- **[NOTED]** For [theseus](https://github.com/facebookresearch/theseus) library, you may build from scratch and add system PATH in ```planner.py```

### Data Process

- Preprocess data for training & testing: 

```
python preprocess.py \
--root_dir path/to/your/Waymo_Dataset/scenario/ \
--save_dir path/to/your/processed_data/ \
--processes=16
```

- You may also refer to [Waymo_candid_list](https://github.com/MCZhi/GameFormer/blob/main/open_loop_planning/waymo_candid_list.csv) for more interactive and safety-critical scenarios filtered in ```scenario/validation```

### Training & Testing

- Train & Eval the model using the command:

```
python -m torch.distributed.launch \
        --nproc_per_node 1 \ # number of gpus
        --master_port 16666 \
        training.py \
        --data_dir path/to/your/processed_data/ \
        --save_dir path/to/save/your/logs/
```

- Conduct Open-loop Testing using the command:

```
python testing.py \
        --data_dir path/to/your/testing_data/ \
        --model_dir path/to/pretrained/model/
```

## Citation
If you find this repository useful for your research, please consider giving us a star &#127775; and citing our paper.

```angular2html
@article{liu2023occupancy,
  title={Occupancy Prediction-Guided Neural Planner for Autonomous Driving},
  author={Liu, Haochen and Huang, Zhiyu and Lv, Chen},
  journal={arXiv preprint arXiv:2305.03303},
  year={2023}
}