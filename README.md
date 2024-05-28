# CXR-CVD-Risk: Deep learning to predict 10-year cardiovascular risk from chest radiographs

[Weiss J, Raghu VK, Paruchuri K, Zinzuwadia AN, Natarajan P, Aerts HJWL, and Lu MT. Deep learning to estimate cardiovascular risk from chest radiographs: A risk prediction study. *Annals of Internal Medicine*. 2024;177(4):409-17. doi:10.1001/jamanetworkopen.2019.7416](https://www.acpjournals.org/doi/10.7326/M23-1898).

## Installation
This inference code was tested on Ubuntu 18.04.3 LTS, conda version 4.8.0, python 3.9.9, fastai 2.5.3, cuda 11.2, pytorch 1.10 and cadene pretrained models 0.7.4. 

Inference can be run on the GPU or CPU, and should work with ~4GB of GPU or CPU RAM. For GPU inference, a CUDA 10 capable GPU is required.

Model weights are available via Dropbox: https://www.dropbox.com/sh/ao93iyib605oq5w/AACuqINPnKF0k4ekMPtUSIExa?dl=0

This example is best run in a conda environment:

```bash
git lfs clone https://github.com/vineet1992/CXR-CV-Risk/
cd location_of_repo
mkdir models
###Download model weights from Dropbox and include them in the "models" directory
conda create -n CXR_CV python=3.9
conda activate CXR_CV
conda install -c fastai fastai ##follow instructions for your OS here: https://github.com/fastai/fastai>
## You may need to install pytorch according to CUDA version and OS first - see https://pytorch.org/get-started/locally/
conda install docopt
pip install pretrainedmodels==0.7.4

python run_cxr_cv_risk.py dummy_datasets/test_images/ path/to/model/weights/PLCO_CV_Risk_010422 output/output.csv
```

Dummy image files are provided in `dummy_datasets/test_images/;`

## Datasets
PLCO (NCT00047385) data used for model development and testing are available from the National Cancer Institute (NCI, https://biometry.nci.nih.gov/cdas/plco/). NLST (NCT01696968) testing data is available from the NCI (https://biometry.nci.nih.gov/cdas/nlst/) and the American College of Radiology Imaging Network (ACRIN, https://www.acrin.org/acrin-nlstbiorepository.aspx). Due to the terms of our data use agreement, we cannot distribute the original data. Please instead obtain the data directly from the NCI and ACRIN.

## Image processing
PLCO radiographs were provided as scanned TIF files by the NCI. TIFs were converted to PNGs with a minimum dimension of 512 pixels with ImageMagick v6.8.9-9. 

Many of the PLCO radiographs were rotated 90 or more degrees. To address this, we developed a CNN to identify rotated radiographs. First, we trained a CNN using the resnet34 architecture to identify synthetically rotated radiographs from the [CXR14 dataset](http://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf). We then fine tuned this CNN using 11,000 manually reviewed PLCO radiographs. The rotated radiographs identified by this CNN in `preprocessing/plco_rotation_github.csv` were then corrected using ImageMagick. 

```bash
cd path_for_PLCO_tifs
mogrify -path destination_for_PLCO_pngs -trim +repage -colorspace RGB -auto-level -depth 8 -resize 512x512^ -format png "*.tif"
cd path_for_PLCO_pngs
while IFS=, read -ra cols; do mogrify -rotate 90 "${cols[0]}"; done < /path_to_repo/preprocessing/plco_rotation_github.csv
```

NLST radiographs were provided as DCM files by ACRIN. We chose to first convert them to TIF using DCMTK v3.6.1, then to PNGs with a minimum dimension of 512 pixels through ImageMagick to maintain consistency with the PLCO radiographs:

```bash
cd path_to_NLST_dcm
for x in *.dcm; do dcmj2pnm -O +ot +G $x "${x%.dcm}".tif; done;
mogrify -path destination_for_NLST_pngs -trim +repage -colorspace RGB -auto-level -depth 8 -resize 512x512^ -format png "*.tif"
```


The orientation of several NLST chest radiographs was manually corrected:

```
cd destination_for_NLST_pngs
mogrify -rotate "90" -flop 204025_CR_2000-01-01_135015_CHEST_CHEST_n1__00000_1.3.51.5146.1829.20030903.1123713.1.png
mogrify -rotate "-90" 208201_CR_2000-01-01_163352_CHEST_CHEST_n1__00000_2.16.840.1.113786.1.306662666.44.51.9597.png
mogrify -flip -flop 208704_CR_2000-01-01_133331_CHEST_CHEST_n1__00000_1.3.51.5146.1829.20030718.1122210.1.png
mogrify -rotate "-90" 215085_CR_2000-01-01_112945_CHEST_CHEST_n1__00000_1.3.51.5146.1829.20030605.1101942.1.png
```

## Acknowledgements
I thank the NCI and ACRIN for access to trial data, as well as the PLCO and NLST participants for their contribution to research. I would also like to thank the fastai and Pytorch communities as well as the National Academy of Medicine for their support of this work. A GPU used for this research was donated as an unrestricted gift through the Nvidia Corporation Academic Program. The statements contained herein are mine alone and do not represent or imply concurrence or endorsements by the above individuals or organizations.


