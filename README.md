Weakly supervised learning of deep metrics for stereo reconstruction (MC-CNN-WS)
===================================================================================

# Citation 

Please cite our [paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Tulyakov_Weakly_Supervised_Learning_ICCV_2017_paper.pdf)
if you use code from this repository in your work.

    @inproceedings{tulyakov2017weakly,
      title={Weakly supervised learning of deep metrics for stereo reconstruction},
      author={Tulyakov, Stepan and Ivanov, Anton and Fleuret, Francois},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
      pages={1339--1348},
      year={2017}
    }

# Installation 

Install Torch, OpenCV 2.4, and png++.

Download [mc-cnn](https://github.com/jzbontar/mc-cnn/blob/master/README.md) and mc-cnn-ws 

    mkdir mc-cnn-ws-experiment
    cd mc-cnn-ws-experiment
    git clone https://github.com/jzbontar/mc-cnn.git
    git clone https://github.com/tlkvstepan/mc-cnn-ws.git
    
Following instruction for [mc-cnn](https://github.com/jzbontar/mc-cnn/blob/master/README.md) compile the code,
download and preprocess dataset. Dont forget to modify make file according to your system. 

Install mc-cnn-ws

    cd mc-cnn-ws
    make 
    ln -s /
    

The training can be performed by running:  

    TRAIN_NET.lua <mode> <method> <arch> <set>

    <mode>   - normal / debug.
    <method> - mil / contrastive / mil-contrastive / contrastive-dp
    <arch>   - fst-kitti / fst-mb
    <set>    - kitti / kitti_ext / kitti15 / kitti15_ext / mb
