### Prepare ONNX model for `scrdf`

ONNX model for scrdf is available to download: [`scrdf_10g.onnx`](https://drive.google.com/file/d/1t4xd9tBTY4AQMSv2hXnaSwHAuZgwV2Ew/view?usp=drive_link).  
Follow the instructions below if for any reason you need to re-generate it from the `model.path` file.


make sure your venv is activated
```bash
source .venv-insightface/bin/activate
```
downgrade numpy:

```bash
pip install numpy==1.21.1
```

Update `site-packages/torch/onnx/utils.py` :

```bash
def _decide_input_format(model, args):
    return args
```

Or just run this command if you are using the same .venv setup as above:

```bash
sed -i '/def _decide_input_format(model, args):/,/return args/{/def _decide_input_format(model, args):/!d}' /home/ubuntu/.venv-insightface/lib/python3.8/site-packages/torch/onnx/utils.py
echo 'def _decide_input_format(model, args):' >> /home/ubuntu/.venv-insightface/lib/python3.8/site-packages/torch/onnx/utils.py
echo '    return args' >> /home/ubuntu/.venv-insightface/lib/python3.8/site-packages/torch/onnx/utils.py
```

Convert to ONNX:


```bash
cd insightface/detection/scrfd
export input_img='/home/ubuntu/insightface/insightface_assets/data/chimp_40/Chimp_40/Chimp_FilmRip_MVP2MostVerticalPrimate.2001.0000.png'
export onnx_fpath='/home/ubuntu/insightface/insightface_assets/models/scrfd_10g.onnx'
python tools/scrfd2onnx.py configs/scrfd/scrfd_10g.py \
	/home/ubuntu/insightface/insightface_assets/models/model.pth \
	--input-img ${input_img} \
    --output-file ${onnx_fpath}
```

Upgrade numpy back in your venv...
```
pip install --upgrade numpy 
```