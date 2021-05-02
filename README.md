# CorDA
Code for our paper [Domain Adaptive Semantic Segmentation with Self-Supervised Depth Estimation](http://arxiv.org/abs/2104.13613) 
<img src="./extra/fig.png" alt="alt text" width="500" height="350">

## Prerequisite
Please create and activate the following conda envrionment 
```bash
# It may take several minutes for conda to solve the environment
conda env create -f environment.yml
conda activate corda 
```
Code was tested on a V100 with 16G Memory.

## Train a CorDA model
```bash
# Train for the SYNTHIA2Cityscapes task
bash run_synthia_stereo.sh
# Train for the GTA2Cityscapes task
bash run_gta.sh
```

## Test the trained model
```bash
bash shells/eval_syn2city.sh
bash shells/eval_gta2city.sh
```
Pre-trained models are provided ([Google Drive](https://drive.google.com/file/d/1yYV5O7In2sgYKA9cY8-12p9VdyWtRuFH/view?usp=sharing)). Please put them in `./checkpoint`. 

+ The provided SYNTHIA2Cityscapes model achieves 56.3 mIoU (16 classes) at the end of the training. 
+ The provided GTA2Cityscapes model achieves 57.7 mIoU (19 classes) at the end of the training.

Reported Results on SYNTHIA2Cityscapes
| Method | mIoU*(13)| mIoU(16) |
| -------- | -------- | -------- |
|  CBST   | 48.9   | 42.6     |
|  FDA    | 52.5   | -        |
|  DADA   | 49.8   | 42.6     |
|  DACS   | 54.8   | 48.3     |
|**CorDA**| **62.8**   | **55.0**     |

## Citation
Please cite our work if you find it useful.
```bibtex
@misc{wang2021domain,
      title={Domain Adaptive Semantic Segmentation with Self-Supervised Depth Estimation}, 
      author={Qin Wang and Dengxin Dai and Lukas Hoyer and Olga Fink and Luc Van Gool},
      year={2021},
      eprint={2104.13613},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement 
+ DACS is used as our codebase and  our DA baseline [official](https://github.com/vikolss/DACS) 
+ SFSU as the source of stereo Cityscapes depth estimation [Official](https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/) 

## Data links
+ Download [links](https://qin.ee/depth/) 
    + Stereo Depth Estimation for Cityscapes
    + Mono Depth Estimation for GTA
+ Dataset Folder Structure [Tree](./extra/data_tree)


For questions regarding the code, please contact wang@qin.ee .
