
## Install dependency and download model weight.

```bash 
conda create -n mfd python=3.10
conda activate mfd
pip install ultralytics

# download with modelscope
cd MFD/
wget -c https://www.modelscope.cn/models/wanderkid/PDF-Extract-Kit/resolve/master/models/MFD/weights.pt

# you can also download with huggingface
# https://huggingface.co/wanderkid/PDF-Extract-Kit/blob/main/models/MFD/weights.pt
```

## Run the demo
```bash
python demo.py
```




