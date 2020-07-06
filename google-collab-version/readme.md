# Pytorch Digital Pathology Collab Version

## Helpful tips

1. It's free.
2. Your computing resources are not used
3. Speed of your computer does not matter.
4. Access (Graphics Processing Unit) &amp; TPU(Tensor Processing Unit)
5. Most libraries come pre-installed, do check the versions if your code needs a specific version.

## Pre-Requisite

1. Need a Google Drive account.
2. On the Google Drive, for this project, below folders will be needed on your drive
```
My Drive
|___PytorchDigitalPathology
    │___classification_lymphoma_densenet
    │___segmentation_epistroma_unet   
    │___visualization_densenet

```
1. Version Requirements
    - scikit_image==0.15.0
    - scikit_learn==0.21.3
    - opencv_python_headless==4.1.1.26
    - scipy==1.3.0
    - torch==1.5.0
    - torchvision==0.6.0
    - numpy==1.16.4
    - umap_learn==0.3.10
    - Pillow==6.2.0
    - tensorboardX==1.9
    - ttach==0.0.2
    - albumentations==0.4.3
    

## Classification Lymphoma Densenet.

For this Project, create on your drive the following folder structure

My Drive --> PytorchDigitalPathology --> classification_lymphoma_densenet

### Digital Pathology Classification Using Pytorch + Densenet

[http://www.andrewjanowczyk.com/digital-pathology-classification-using-pytorch-densenet/](http://www.andrewjanowczyk.com/digital-pathology-classification-using-pytorch-densenet/)

### Retrieve the Lymphoma data set and untar it in the Google Drive folder

google-collab-version/PytorchDigitalPathology/classification_lymphoma_densenet/RetrieveLymphomaData.ipynb

### Making the Training and Testing Databases

Make HDF5 files, using the files retrieved earlier.

google-collab-version/PytorchDigitalPathology/classification_lymphoma_densenet/make_hdf5.ipynb

Changes Made
- Change 1 - making a connection to the google drive so that the data files can be accessed.
```python
from google.colab import drive

drive.mount('/gdrive')

%cd /gdrive/My Drive/PytorchDigitalPathology/classification_lymphoma_densenet/data/
```
- Change 2 - Added code to the file so that the .pytable files will be stored back in the google drive. This will be helpful to access them using the other Colab Notebooks.

### Training a Model

Train a Densenet Model using Albumentations

google-collab-version/PytorchDigitalPathology/classification_lymphoma_densenet/train_densenet_albumentations.ipynb

**Please check the Runtime and change it to GPU if not already set.**

Changes Made

- Change 1 - making a connection to the google drive so that the data files can be accessed.
```python
from google.colab import drive

drive.mount('/gdrive')

%cd /gdrive/My Drive/PytorchDigitalPathology/classification_lymphoma_densenet/data/
```
- Change 2 -  this library does not come pre-installed.
```python
!pip install tensorboardX
```
- Change 3 -  we need to use this specific version of albumnetations. The one that comes pre-installed is very old.
```python
! pip install albumentations==0.4.5
```
- Change 4 - We have used version 2.2.0
```python
import tensorflow as tf
print(tf.__version__) 
```

- Change 5 - This command helps load the tensorboard ext.
```python
%load_ext tensorboard 
```
- Change 6 - Post running the epocs - use this command.This command helps show the tensorboard in the colab notebook.

```python
%tensorboard --logdir runs
```

- Change 7 - Post execution stores the model.pth file back on the google drive location.

### Visualizing results in the validation set

Visualize Validation Results.

google-collab-version/PytorchDigitalPathology/classification_lymphoma_densenet/visualize_validation_results.ipynb

**Please check the Runtime and change it to GPU if not already set.**

Changes Made

- Change 1 - making a connection to the google drive so that the data files can be accessed.
```python
from google.colab import drive

drive.mount('/gdrive')

%cd /gdrive/My Drive/PytorchDigitalPathology/classification_lymphoma_densenet/data/
```
- Change 2 -  this library does not come pre-installed.
```python
!pip install tensorboardX
```
- Change 3 -  we need to use this specific version of albumnetations.
```python
! pip install albumentations==0.4.5
```
- Change 4 - We have used version 2.2.0
```python
import tensorflow as tf

print(tf.__version__) 
```

- Change 5 - This command helps load the tensorboard ext.
```python
 %load_ext tensorboard
```
- Change 6 - Post running the epocs - use this command. This command helps show the tensorboard in the colab notebook.
```python
%tensorboard --logdir runs
```
### Generating Output

Make Ouput Densenet colab Notebook.

google-collab-version/PytorchDigitalPathology/classification_lymphoma_densenet/make_output_densenet_cmd.ipynb

**Please check the Runtime and change it to GPU if not already set.**

Changes Made

- Change 1 - Check for the versions of all the libraries used.
- Change 2 - making a connection to the google drive(will be used to access data file, and write back the output.)
```python
from google.colab import drive

drive.mount('/gdrive')

%cd /gdrive/My Drive/PytorchDigitalPathology/classification_lymphoma_densenet/data/
```
- Change 3 - Changed the args parser to below
```python
args = parser.parse_args(["*.tif","-p256","-m/gdrive/My Drive/PytorchDigitalPathology/classification_lymphoma_densenet/
/data/lymphoma_densenet_best_model.pth","-b/gdrive/My Drive/PytorchDigitalPathology/classification_lymphoma_densenet/data/**"] )

```
- Change 4 -

Added "/"
```python
files = glob.glob(args.basepath + "/" + args.input\_pattern[0])
```
- Change 5 -
```
output = np.zeros((0,checkpoint["n_classes"]))

To output = np.zeros((0,checkpoint["num_classes"]))
```
## Segmentation Epistroma Unet.

For this Project, create on your drive the following folder structure

My Drive → PytorchDigitalPathology  segmentation\_epistroma\_unet

### Digital Pathology Segmentation Using Pytorch + Unet

[http://www.andrewjanowczyk.com/pytorch-unet-for-digital-pathology-segmentation/#db](http://www.andrewjanowczyk.com/pytorch-unet-for-digital-pathology-segmentation/#db)

### Retrieve the Epistroma data set and untar it in the Google Drive folder

google-collab-version/PytorchDigitalPathology/segmentation_epistroma_unet/RetrieveEpistromaData.ipynb

### Making the Training and Testing Databases

Make HDF5 files, using the files retrieved earlier.

google-collab-version/PytorchDigitalPathology/segmentation_epistroma_unet/make_hdf5.ipynb

Changes Made

- Change 1 - making a connection to the google drive so that the data files can be accessed.
```python
from google.colab import drive

drive.mount('/gdrive')

%cd /gdrive/My Drive/PytorchDigitalPathology/segmentation_epistroma_unet/data
```
- Change 2 - Added code to the file so that the .pytable files will be stored back in the google drive. This will be helpful to access them using the other Colab Notebooks.

### Training a Model

Train a Unet Model using Albumentations

google-collab-version/PytorchDigitalPathology/segmentation_epistroma_unet/train_unet_albumentations.ipynb

**Please check the Runtime and change it to GPU if not already set.**

Changes Made

- Change 1 - making a connection to the google drive so that the data files can be accessed.
```python
from google.colab import drive

drive.mount('/gdrive')
```
- Change 2 - Copy the unet.py code to the Colab Notebook Runtime
```python
!cp '/gdrive/My Drive/PytorchDigitalPathology/segmentation_epistroma_unet/unet.py' unet.py
```
- Change 3 - 
```python
%cd /gdrive/My\ Drive/segmentation\_epistroma\_unet/data
```
- Change 4 - this library does not come pre-installed.
```python
!pip install tensorboardX 
```
- Change 5 - we need to use this specific version of albumnetations.
```python
! pip install albumentations==0.4.5
```
- Change 6 - Put the unet.py in the sys.path and import
```python
import sys

sys.path.append(&#39;/content&#39;)

from unet import UNet #code borrowed from https://github.com/jvanvugt/pytorch-unet
```
- Change 7 - import tensorflow as tf
```python
print(tf.__version__) #We have used version 2.2.0
```

- Change 8 - This command helps load the tensorboard ext.
```python
%load_ext tensorboard
```
- Change 9 - Post running the epocs - use this command. This command helps show the tensorboard in the colab notebook.
```python
%tensorboard --logdir runs
```
- Change 10 - Post execution stores the model.pth file back on the google drive data location.

### Visualizing results in the validation set

Visualize Validation Results.

google-collab-version/PytorchDigitalPathology/segmentation_epistroma_unet/visualize_validation_results.ipynb

**Please check the Runtime and change it to GPU if not already set.**

Changes Made

- Change 1 - making a connection to the google drive so that the data files can be accessed.
```python
from google.colab import drive

drive.mount('/gdrive')
```
- Change 2 - Copy the unet.py code to the Colab Notebook Runtime
```python
!cp '/gdrive/My Drive/PytorchDigitalPathology/segmentation_epistroma_unet/unet.py' unet.py
```
- Change 3 
```python
%cd /gdrive/My\ Drive/PytorchDigitalPathology/segmentation\_epistroma\_unet/data
```
- Change 4 - Put the unet.py in the sys.path and import
```python
import sys

sys.path.append('/content')

from unet import UNet #code borrowed from https://github.com/jvanvugt/pytorch-unet
```
- Change 5 - !pip install tensorboardX , as this library does not come pre-installed.

### Generating Output

Make Ouput Unet colab Notebook.

google-collab-version/PytorchDigitalPathology/segmentation_epistroma_unet/make_output_unet_cmd.ipynb

**Please check the Runtime and change it to GPU if not already set.**

Changes Made

- Change 1 - Check for the versions of all the libraries used.
- Change 2 - making a connection to the google drive so that the data files can be accessed.
```python
from google.colab import drive

drive.mount('/gdrive')
```
- Change 3 - Copy the unet.py code to the Colab Notebook Runtime
```python
!cp '/gdrive/My Drive/PytorchDigitalPathology/segmentation_epistroma_unet/unet.py' unet.py
```
- Change 4 - Put the unet.py in the sys.path and import
```python
import sys

sys.path.append('/content')

from unet import UNet #code borrowed from [https://github.com/jvanvugt/pytorch-unet](https://github.com/jvanvugt/pytorch-unet)
```
- Change 5 - 
```python
%cd /gdrive/My\ Drive/PytorchDigitalPathology/segmentation\_epistroma\_unet/data
```
- Change 6 - Changed the args parser to below
```python
args = parser.parse_args(["*.png","-o/gdrive/My Drive/PytorchDigitalPathology/segmentation_epistroma_unet/data/output","-m/gdrive/My Drive/PytorchDigitalPathology/segmentation_epistroma_unet/data/epistroma_unet_best_model.pth","-b/gdrive/My Drive/PytorchDigitalPathology/segmentation_epistroma_unet/data/masks"] )
```
- Change 7 -

Added "/"
```python
files = glob.glob(args.basepath + "/" + args.input\_pattern[0])
```

## Visualization Densenet

For this Project, create on your drive the following folder structure

My Drive →PytorchDigitalPathology → visualization_densenet

### Visualization Densenet using Pytorch

[http://www.andrewjanowczyk.com/visualizing-densenet-using-pytorch/](http://www.andrewjanowczyk.com/visualizing-densenet-using-pytorch/)

### Making the Training and Testing Databases

Make HDF5 files,

google-collab-version/PytorchDigitalPathology/visualization_densenet/make_hdf5_synthetic_circles_and_boxes.ipynb

Changes Made

- Change 1 - making a connection to the google drive, and navigate to the folder where data files will be written at.
```python
from google.colab import drive

drive.mount("gdrive")

%cd /gdrive/My Drive/PytorchDigitalPathology/visualization_densenet

```
- Change 2 - Added code to the file so that the .pytable files will be stored back in the google drive. This will be helpful to access them using the other Colab Notebooks.

### Training a Model

Train a Densenet Model

google-collab-version/PytorchDigitalPathology/visualization_densenet/train_densenet.ipynb

**Please check the Runtime and change it to GPU if not already set.**

Changes Made

- Change 1 - making a connection to the google drive so that the data files can be accessed.
```python
from google.colab import drive

drive.mount('/gdrive')

%cd /gdrive/My Drive/PytorchDigitalPathology/visualization\_densenet
```
- Change 2 -  this library does not come pre-installed.
```python
!pip install tensorboardX
```
- Change 3 
```python
!pip install line_profiler
```
- Change 4 - 
```python
import tensorflow as tf

print(tf.\_\_version\_\_) # We have used version 2.2.0
```

- Change 5 - This command helps load the tensorboard ext.
```python
%load_ext tensorboard
```
- Change 6 - Post running the epocs - use this command. This command helps show the tensorboard in the colab notebook.
```python
%tensorboard --logdir runs
```
- Change 7 - Post execution stores the model.pth file back on the google drive location.

### Visualizing results

Visualize Validation Results.

google-collab-version/PytorchDigitalPathology/visualization_densenet/densenet_visualization_notebook.ipynb

**Please check the Runtime and change it to GPU if not already set.**

Changes Made

- Change 1 - making a connection to the google drive so that the data files can be accessed.
```python
from google.colab import drive

drive.mount('/gdrive')

%cd /gdrive/My Drive/PytorchDigitalPathology/PytorchDigitalPathology/visualization\_densenet
```
- Change 2 - 
```python
!cp -r 'pytorch-cnn-visualizations' /content
```
- Change 3 - configure the code in the sys.path
```python
import sys

sys.path.append('/content/pytorch-cnn-visualizations/src/')
```
