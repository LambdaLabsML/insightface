## Fresh instance workflow

### Setup insightFace

Create script `install.sh`

```bash
#!/bin/bash
sudo apt-get install -y python3-pybind11
virtualenv -p /usr/bin/python3.8 .venv-insightface
. .venv-insightface/bin/activate
pip install --upgrade pip \
    torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116 \
    -U openmim \
    insightface==0.6.2 mmcv-full==1.5.0

git clone https://github.com/LambdaLabsML/insightface.git
cd insightface/detection/scrfd
pip install -r requirements/build.txt
pip install -v -e .
pip install -r requirements_add.txt
```

Run install script:

```bash
sudo chmod +x install.sh
./install.sh
```

### Setup data

```bash
storage
|-- insightface_assets
|		|-- data
|		|   |-- chimp_50
|		|   |-- chimp_2000_0
|		|-- models
|				|-- synthetic_resnet50d.ckpt (ldkms detector)
|               |-- scrfd_10g.onnx (bbox detector)
|               |-- model.path (only use if you need to re-generate the onnx file for some reason)
|-- .insightface-venv
|-- insightface
```

Data sets:
- `chimp_40` (ask DV for datasets)

Models
- `scrfd_10g.onnx`[[download](https://drive.google.com/file/d/1t4xd9tBTY4AQMSv2hXnaSwHAuZgwV2Ew/view?usp=sharing)]
- `synthetic_resnet50d.ckpt` [[download](https://www.notion.so/Streamlining-InsightFace-workflow-1125ab65c04849fbab1c5bc1ca64274f)]

If for some reason you need to re-generate the onnx file for the scrdf_10g bbox model, refer to [this guide](PrepareONNX.md)
The model to convert to onnx is available here: [`model.path`](https://onedrive.live.com/?authkey=%21AArBOLBe%5FaRpryg&id=4A83B6B633B029CC%215541&cid=4A83B6B633B029CC)

### Inference

Make sure environment is activated

```bash
source .venv-insightface/bin/activate
```

```bash
export root_dir="/home/ubuntu/insightface"
export models_dir="${root_dir}/insightface_assets/models"
export img_dir="${root_dir}/insightface_assets/data/chimp_40"


# Step 1: bbox (required)
python "${root_dir}/insightface/examples/chimp/create_data.py" \
--stage bbox \
--bbox-method scrfd \
--input_image_dir "${img_dir}" \
--bbox_detector_path "${models_dir}/scrfd_10g.onnx"

# Step 2: ldmks
python "${root_dir}/insightface/examples/chimp/create_data.py" \
--stage ldmks \
--input_image_dir "${img_dir}" \
--ldmks_detector_path "${models_dir}/synthetic_resnet50d.ckpt" \
--bbox_detector_path "${models_dir}/scrfd_10g.onnx" # ldkms should run bbox detection automatically but it doesnt ?
```

### Finetuning

Make sure environment is activated
```bash
source .venv-insightface/bin/activate
```


Prepare a training dataset; for instance
```bash
storage
|-- insightface_assets
		|-- data
		|   |-- chimp_50
            |-- 0_bbox.txt
            |-- 0_ldkms.txt
            |-- 0_img.png

```

Finetuning:
```bash
export img_dir=/home/ubuntu/insightface/insightface_assets/data/chimp_40_scrfd_10g_0.3_1.5
export model_dir=/home/ubuntu/insightface/insightface_assets/models

python trainer_synthetics.py \
--batch_size 8 \
--root "${img_dir}" \
--pre-trained-path "${model_dir}/synthetic_resnet50d.ckpt" \
--output-ckpt-path "${model_dir}/synthetic_resnet50d_ft0.ckpt" \
--early_stopping True \
--num-epochs 1000 \
--lr 0.00001 \
--num-gpus 1
```