#  Character Detection Matching (CDM)

## Demo

CDM: A Reliable Metric for Fair and Accurate Formula Recognition

![demo](assets/demo/demo.png)

Compair with BLEU and ExpRate:

![demo](assets/demo/cases.png)

## Installation Guide

Nodejs, imagemagic, pdflatex are requried packages when render pdf files and convert them to images, here are installation guides.

### install nodejs

download the package from [offical website](https://registry.npmmirror.com/binary.html?path=node/latest-v16.x/), and then run these commands.
```
tar -xvf node-v16.13.1-linux-x64.tar.gz

mv node-v16.13.1-linux-x64/* /usr/local/nodejs/

ln -s /usr/local/nodejs/bin/node /usr/local/bin

ln -s /usr/local/nodejs/bin/npm /usr/local/bin

node -v
```

### install imagemagic

the version of imagemagic installed by `apt-gt` usually be 6.x, so we also install it from source code.
```
git clone https://github.com/ImageMagick/ImageMagick.git ImageMagick-7.1.1

cd ImageMagick-7.1.1

./configure

make

sudo make install

sudo ldconfig /usr/local/lib

convert --version
```

### install latexpdf

```
apt-get update

sudo apt-get install texlive-full
```

### install python requriements

```
pip install -r requirements.txt
```


## Usage

Should the installation go well, you may now enjoy the evaluation pipeline.

### 1. batch evaluation 

```
python evaluation.py -i {path_to_your_input_json}
```

the format of input json like this:
```
[
    {
        "img_id": "case_1",      # optional key
        "gt": "y = 2z + 3x",
        "pred": "y = 2x + 3z"
    },
    {
        "img_id": "case_2",
        "gt": "y = x^2 + 1",
        "pred": "y = x^2 + 1"
    },
    ...
]
```

### 2. run a gradio demo

```
python app.py
```