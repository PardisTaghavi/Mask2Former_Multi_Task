for personal use - please refer to original rep.

data to download

```https://drive.google.com/file/d/1rzaJzFzqIuPeYtEI26LEHyRm5yOTfFN/view?usp=sharing```

installation

```
conda create --name mask2former python=3.8 -y
conda activate mask2former
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install -U opencv-python
git clone https://github.com/PardisTaghavi/Mask2Former_Multi_Task.git
cd Mask2Former_Multi_Task

git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git

cd ..
pip install -r requirements.txt
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```

```
export DETECTRON2_DATASETS=/path/to/datasets

python train_net.py --num-gpus 2 --config-file configs/cityscapes/instance-segmentation/dinov2/maskformer2_dino_small_bs16_90k.yaml 

```

