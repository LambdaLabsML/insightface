# Install

```
git clone https://github.com/chuanli11/insightface.git

cd insightface/

virtualenv -p /usr/bin/python3.8 venv

pip install -r requirements.txt

cd python-package/
pip install -e .

# For running the synthetic example, you need
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install pytorch_lightning
```


# Convert insightface face alignment model to onnx

```
python examples/torch2onnx.py <path-to-ckpt>
```

Results will be saved in `insightface/examples/onnx`
