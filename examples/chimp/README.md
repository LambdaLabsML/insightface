# Finetune landmark detector with a custom dataset

There are two steps for getting facial landmarks from an image:

- face detection
- landmark detection

The first step detects bounding boxes that "cut" the face out of the input image. The second step detects the actual landmarks from the cutout region.

There are multiple face detectors and landmark detectors in this insightface library. We choose to use the `scrfd_10g` face detector and the `alignment/synthetics` landmark detector.

## Installation

Running the `scrfd` face detector requires some extra dependencies which adds constraint to CDUA/PyTorch version. Here are a couple of verified ways to set up the environment:

**Setup insightface on a CUDA 11.6-11.8 machine**

In this case you can directly use the system-wise installed CUDA. Here is how to create an virtualenv that works:

```
sudo apt-get install python3-pybind11

virtualenv -p /usr/bin/python3.8 .venv-insightface
. .venv-insightface/bin/activate
pip install --upgrade pip

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

pip install -U openmim
mim install mmcv-full==1.5.0

git clone https://github.com/LambdaLabsML/insightface.git
cd insightface/detection/scrfd

pip install -r requirements/build.txt
pip install -v -e .
pip install -r requirements_add.txt
```

**Setup insightface on a CUDA 12.2 machine**

CUDA 12.2 is too new for this. A walk-around is to create a conda environment (tested with conda 4.12.0)

Fist, install miniconda

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
conda init bash
source ~/.bashrc
```

Verify conda install:

```
conda --version
```

Environment setup:

```
sudo apt-get install python3-pybind11

conda create -n insightface python=3.8.10
conda activate insightface
conda install cudatoolkit=11.6
conda install -c "nvidia/label/cuda-11.6.0" cuda-nvcc

pip install --upgrade pip
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

# for rich to be found
export PYTHONPATH=$PYTHONPATH:/home/ubuntu/anaconda3/envs/insightface/lib/python3.8/site-packages

pip install -U openmim
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12/index.html

git clone https://github.com/LambdaLabsML/insightface.git
cd insightface/detection/scrfd

pip install -r requirements/build.txt
pip install -v -e .
pip install -r requirements_add.txt
```

## Model and Dataset Preparation

### Model Preparation

The `scrdf` face detector can be downloaded from [here](https://onedrive.live.com/?authkey=%21AArBOLBe%5FaRpryg&id=4A83B6B633B029CC%215541&cid=4A83B6B633B029CC).

The synthetic landmark detector can be downloaded from [here](https://drive.google.com/file/d/1kNP7qEl3AYNbaHFUg_ZiyRB1CtfDWXR4/view).

For the `scrfd` face detector, we need to convert it to onnx model. To do so, you have to make a change to `site-packages/torch/onnx/utils.py` by letting the `_decide_input_format` function directly return `args`

```
def _decide_input_format(model, args):
    return args
```

Then you can convert the downloaded `SCRFD_10G.pth` to an onnx file with the following command:

```
cd insightface/detection/scrfd
python tools/scrfd2onnx.py configs/scrfd/scrfd_10g.py path-to-SCRFD_10G.pth
```

### Download data and models

Put the input images and models somewhere on your machine, e.g.

```
input_image_dir: /home/ubuntu/Chimp/datasets/Chimp_40
bbox_detector_path: /home/ubuntu/Chimp/models/scrfd/scrfd_10g.onnx
ldmks_detector_path: /home/ubuntu/Chimp/models/synthetic_resnet50d.ckpt
```

## Create a Refined Landmark Dataset

The basic idea is to

- Run the out-of-box face detection and landmark detection
- Pick the "good" result to finetune the landmark detection

### Face detection

```
python create_data.py \
--stage bbox --bbox-method scrfd \
--input_image_dir /home/ubuntu/Chimp/datasets/Chimp_40 \
--bbox_detector_path /home/ubuntu/Chimp/models/scrfd/scrfd_10g.onnx
```

The output will be saved as `_bbox.txt` files in `<input_image_dir>_<detetor_name>_<bbox_confidence>_<bbox_size_scale>`. For example, `Chimp_40_scrfd_10g_0.3_2.5/image00001_bbox.txt`.

### Landmark detection

```
python create_data.py \
--stage ldmks \
--input_image_dir /home/ubuntu/Chimp/datasets/Chimp_40 \
--ldmks_detector_path /home/ubuntu/Chimp/models/synthetic_resnet50d.ckpt \
--dataset_postfix _dataset
```

The output will be saved as `_ldmks.txt` files in
`<input_image_dir>_<detetor_name>_<bbox_confidence>_<bbox_size_scale>`. In this example, `Chimp_40_scrfd_10g_0.3_2.5/image00001_ldmks.txt`

This step will also save the rendered landmarks in a different folder `<input_image_dir>_<detetor_name>_<bbox_confidence>_<bbox_size_scale>_render`. In this example, `Chimp_40_scrfd_10g_0.3_2.5_render`.

It will also create an empty "dataset" folder for a dataset of high-quality detections `<input_image_dir>_<detetor_name>_<bbox_confidence>_<bbox_size_scale>_dataset`. In this example,
`Chimp_40_scrfd_10g_0.3_2.5_dataset`.

### Select High-quality landmarks and create a dataset for finetune

**You need to manually copy the high-quality landmark renderings to the "dataset" folder**. Then use the following command to copy the detection metadata (`_bbox.txt` and `_ldmks.txt`) into it, and create a [pickle file](https://github.com/deepinsight/insightface/blob/master/alignment/synthetics/tools/prepare_synthetics.py#L68-L69) for finetuning the landmark detector.

```
python create_data.py --stage dataset \
--input_image_dir /home/ubuntu/Chimp/datasets/Chimp_40 \
--dataset_postfix _dataset
```

## Finetune

Use the following command to finetune the `synthetic_resnet50d` landmark detector using the dataset prepared above.

```
python trainer_synthetics.py --batch_size 8 \
--root /home/ubuntu/Chimp/datasets/Chimp_40_scrfd_10g_0.3_2.5_dataset \
--pre-trained-path /home/ubuntu/Chimp/models/synthetic_resnet50d.ckpt \
--output-ckpt-path /home/ubuntu/Chimp/experiments/synthetics \
--num-epochs 10 \
--lr 0.00001 \
--num-gpus 1
```

## run test

```
cp -r /home/ubuntu/Chimp/datasets/Chimp_40 /home/ubuntu/Chimp/datasets/Chimp_40_cp && \
cp -r /home/ubuntu/Chimp/datasets/Chimp_40_scrfd_10g_0.3_2.5 /home/ubuntu/Chimp/datasets/Chimp_40_cp_scrfd_10g_0.3_2.5 && \
python create_data.py \
--stage ldmks \
--input_image_dir /home/ubuntu/Chimp/datasets/Chimp_40_cp \
--ldmks_detector_path /home/ubuntu/Chimp/experiments/synthetics/<last-ckpt>
```

## Convert to onnx

```
cd insightface/examples && \
python torch2onnx.py \
/home/ubuntu/Chimp/experiments/synthetics/<last-ckpt> \
--output /home/ubuntu/Chimp/models
```
