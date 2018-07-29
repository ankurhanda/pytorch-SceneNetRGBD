# What does this repository contain

* This repository contains the weights of UNet models trained on RGB as well as RGB-D data of [SceneNet RGB-D dataset](https://robotvault.bitbucket.io/scenenet-rgbd.html).

* It has code to reproduce the UNet used in the paper and also provides segmentation evaluation scripts.

* The `test_models.py` contains the code to reproduce the numbers as obtained in the [ICCV 2017 paper](http://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/dyson-robotics-lab/jmccormac_etal_iccv2017.pdf).

# Important things to keep in mind before using the code

* Download the pytorch models from the [google drive link](https://drive.google.com/open?id=1cv95981C8vJ9YZY4QowcqcaU1hW2lj1W). It is 5.8 GBs in total in size.

* This code was converted from the torch implementation used in the paper. The image reader in torch is different from tthe pytorch version and therefore we provide the rgb and depth files convereted from torch in `npy` format. 

* The depth scaling used for `NYUv2` was `1/1000` and `SUN RGB-D` was `1/10000`. This means that if you are using the `NYUv2` pretrained SceneNet RGB-D model that was fine-tuned on `NYUv2` dataset then you should scale down the depth values by a factor of 1000 before using it for any new future experiments and 10000 for SUN RGB-D.

* To obtain the numbers in the paper for 13 class segmentations do `python test_models.py`

# Updates 

* Any future updates will be posted here.
