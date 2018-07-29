# What does this repository contain

* This repository contains the weights of UNet models trained on RGB as well as RGB-D data of SceneNet RGB-D dataset.
* It has code to reproduce the UNet used in the paper and also provides segmentation evaluation scripts.
* The `test_models.py` contains the code to reproduce the numbers as obtained in the ICCV 2017 paper.

# Important things to keep in mind before using the code

* Download the pytorch models from the [google drive link](https://drive.google.com/open?id=1cv95981C8vJ9YZY4QowcqcaU1hW2lj1W)

* This code was converted from the torch implementation used in the paper. The image reader in torch is different from tthe pytorch version and therefore we provide the rgb and depth files convereted from torch in `npy` format. 

* The depth scaling used for `NYUv2` was `1/100` and `SUN RGB-D` was `1/1000`.

* To obtain the numbers in the paper for 13 class segmentations do `python test_models.py`

# Updates 

* Any future updates will be posted here.
