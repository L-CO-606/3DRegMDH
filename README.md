# MDH-Net: advancing 3D brain MRI registration with multi-stage transformer and dual-stream feature refinement hybrid network

**Keywords：Deformable image registration, Medical image registration, Dual-stream network, Brain MRI**

# Dataset
### The OASIS Dataset
This dataset has been graciously provided by Andrew Hoopes and Adrian V. Dalca for the HyperMorph paper mentioned below. 
Available at [https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md].
When utilizing this dataset, kindly acknowledge its use by citing the referenced paper and adhering to 
[the OASIS Data Use Agreement](http://oasis-brains.org/#access).

- **HyperMorph: Amortized Hyperparameter Learning for Image Registration**
  - Authors: A. Hoopes, M. Hoffmann, B. Fischl, J. Guttag, A.V. Dalca
  - Conference: IPMI 2021
  - [https://doi.org/10.1007/978-3-030-78191-0_1](insert_link_here)

- **Open Access Series of Imaging Studies (OASIS): Cross-Sectional MRI Data in Young, Middle Aged, Nondemented, and Demented Older Adults**
  - Authors: D.S. Marcus, T.H. Wang, J. Parker, J.G. Csernansky, J.C. Morris, R.L. Buckner
  - Journal: Journal of Cognitive Neuroscience, 19, 1498-1507.
  - [https://doi.org/10.1162/jocn.2007.19.9.1498](insert_link_here)

### The IXI Dataset
We utilized the preprocessed version of the IXI dataset, curated by [Junyu Chen](https://github.com/junyuchen245) using Freesurfer for standard MRI preprocessing. 
This version of the dataset can be accessed [Google Drive](https://drive.google.com/uc?export=download&id=1-VQewCVNj5eTtc3eQGhTM2yXBQmgm8Ol). 
The preprocessed IXI dataset is made available under [the Creative Commons Attribution-ShareAlike 3.0 Unported License](http://creativecommons.org/licenses/by-sa/3.0/).

If you use this dataset, kindly acknowledge the source of the IXI data: [IXI Dataset - Brain Development](https://brain-development.org/ixi-dataset/), and acknowledge the TransMorph paper:
- **TransMorph: Transformer for unsupervised medical image registration**
  - Authors: Junyu Chen, Eric C. Frey, Yufan He, William P. Segars, Ye Li, Yong Du
  - Journal: Medical Image Analysis, 102615, 1361-8415.
  - [https://doi.org/10.1016/j.media.2022.102615](insert_link_here)


# Implementation
### Train
To train the model, follow these steps:

1. Set the following parameters in `train.py`:
    - Learning rate (lr): 0.0001
    - Head dimension (head_dim): 6
    - Number of heads (num_heads): [8, 4, 2, 1, 1]
    - Image size (img_size): (160, 192, 224)


2. Execute `train.py` to commence training:
```python
python train.py
```
### Test
To perform testing using the same parameters as training, follow these steps:

1. Ensure the parameters in `infer.py` are set to the same values as used in training.


2. Execute `infer.py` to compute the Dice Similarity Coefficient (DSC) score for each image across each anatomical structure. 
```python
python infer.py
```

3. The results will be saved in a `.csv` file located in the `Quantitative_Results` directory, facilitating further analysis.


## Reference
The overall framework and certain components of this codebase are primarily based on the following repositories. We express our sincere gratitude for their prior work and contributions:

<a href="https://github.com/voxelmorph/voxelmorph">VoxelMorph</a>,
<a href="https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration">TransMorph</a>,
<a href="https://github.com/ZAX130/SmileCode">ModeT</a>,
<a href="https://github.com/cwmok/Fast-Symmetric-Diffeomorphic-Image-Registration-with-Convolutional-Neural-Networks">SYM-Net</a>,
<a href="https://github.com/xi-jia/Fourier-Net">Fourier-Net</a>
and
<a href="https://github.com/cwmok/LapIRN">LapIRN</a>.

