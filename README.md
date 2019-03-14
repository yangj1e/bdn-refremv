# bdn-refremv
Deep Bidirectional Estimation for Single Image Reflection Removal. This package is the implementation of the paper:

<small>*[Seeing Deeply and Bidirectionally: A Deep Learning Approach for Single Image Reflection Removal](http://openaccess.thecvf.com/content_ECCV_2018/papers/Jie_Yang_Seeing_Deeply_and_ECCV_2018_paper.pdf)  
[Jie Yang](https://github.com/yangj1e)\*, [Dong Gong](https://donggong1.github.io)\*, [Lingqiao Liu](https://sites.google.com/site/lingqiaoliu83/), [Qinfeng Shi](https://cs.adelaide.edu.au/~javen/index.html).  
In European Conference on Computer Vision (ECCV), 2018.*  (* Equal contribution)
</small>

<img src="imgs/overview.jpg">


## Requirements

+ Python packages
    ```
    pytorch>=0.4.0
    numpy
    pillow
    ```
+ An NVIDIA GPU and CUDA 9.0 or higher

### Conda environment

A minimal conda environment for running the test.sh is provided.

```
conda env create -f env.yml
```

## Usage

+ Download our pretrained model [here](https://drive.google.com/open?id=1zBCl2qI_fT3CwPZkVvZEv37bDIlhakF6). Unpack the archive into `model` folder.

+ Put test images into `samples` folder, and run script `bash test.sh`.

## Examples and Real-world Testing Images
Two examples (on real-world images taken by a mobile phone) are shown in the following: from left to right: I (observed image with reflection), B (recovered reflection-free image) and R (the intermediate reflection image). Please see details and examples in our paper. 

More real-world reflection images can be found in `/samples` for testing.

<p float="left">
    <img src="samples/0001.jpg" width="200">
    <img src="output/B_0001.png" width="200">
    <img src="output/R_0001.png" width="200">
</p>

<p float="left">
<img src="samples/0002.jpg" width="200">
<img src="output/B_0002.png" width="200">
<img src="output/R_0002.png" width="200">
</p>

## Datasets

The synthetic datasets used for training and testing in our paper: 

+ [Training data](https://drive.google.com/open?id=1bbWsGG1qQgB-sbktI2h5vO8UhD1uHaj7)
+ [Test data](https://drive.google.com/open?id=1ZeeKJVbZ_bifsdpAlbguDleViDA4QjCw)


## Citation
If you use this code for your research, please cite our paper:
````
@inproceedings{eccv18refrmv,
  title={Seeing deeply and bidirectionally: a deep learning approach for single image reflection removal},
  author={Yang, Jie and Gong, Dong and Liu, Lingqiao and Shi, Qinfeng},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={654--669},
  year={2018}
}
````



