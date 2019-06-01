# SRCNN in Tensorflow

Tensorflow implementation of **Image Super-Resolution Using Deep Convolutional Networks**.

![intro][intro]

## Implementation Details

Our implementation uses Tensorflow to train SRCNN. We use almost same methods as described in the paper, but there are some slightly different approaches. We train the network with 291 dataset and validate with 91 dataset. Because 291 dataset generates above 25,000 sub-images, we decrease epoch to 3000. At test time, to get as same size as bicubic interpolated image, we fill in cropped pixels on border with bicubic interpolated pixels.

Note that we just train and test with 1-channel. If you want to use with 3-channels, you may add some type-casting code.

## Installation

```bash
git clone https://github.com/jinsuyoo/SRCNN-Tensorflow.git
```

## Requirements

You will need the following to run the above:
- Tensorflow-gpu
- Python3, Numpy, SciPy, imageio, h5py, tqdm

To install quickly, use `requirements.txt`. Example usage:
```bash
pip install -r requirements.txt
```
Note that we run the code with Windows 10, Tensorflow-gpu 1.13.1, CUDA 10.0, cuDNN v7.6.0 

## Documentation

### Training SRCNN
Use `main.py` to train the network. Run `python main.py` to view the training process. Training takes 12-13 hours on a NVIDIA GeForce GTX 1050. Example usage:
```bash
# Quick training
python main.py

# Example usage
python main.py --train_dataset=YOUR_DATASET \
    --valid_dataset=YOUR_DATASET \
    --use_pretrained=False \
    --epoch=1000 \
    --scale=4 \
```

### Testing SRCNN
Also use `main.py` to test the network. Pretrained-model is given. Example usage:
```bash
# Quick testing
python main.py --is_training=False \
    --use_pretrained=True

# Example usage
python main.py --is_training=False \
    --test_dataset=YOUR_DATASET \
    --scale=4
```
  
## Results

### Here are some results testing with various datasets.

![results1][results1]

### Some of the result images

![results2][results2]

## References

- [Official Website][1]
    - I referred to the original Matlab and Caffe code.

- [tegg89/SRCNN-Tensorflow][2]
    - I highly followed the structure of this repository.

[intro]: ./figs/1.png
[results1]: ./figs/2.png
[results2]: ./figs/3.png
[1]: http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html
[2]: https://github.com/tegg89/SRCNN-Tensorflow