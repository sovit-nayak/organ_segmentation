

# Organ Segmentation Using MONAI and TCIA Data

This repository demonstrates a workflow to:
1. Download CT imaging data from [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/)
2. Preprocess medical imaging data using [MONAI](https://monai.io/) transforms.
3. Perform organ segmentation using a pre-trained whole-body segmentation model.
4. Visualize and analyze the resulting organ masks.

**Key Features:**
- Automatic download and loading of DICOM CT datasets from TCIA.
- Easy-to-use preprocessing pipelines that leverage MONAI transforms.
- Integration with a pre-trained MONAI model for whole-body organ segmentation.
- Visualization of data and results using `matplotlib`.
- Computation of organ-specific metrics (e.g., volumes).

---

## Contents

- [Prerequisites](#prerequisites)
- [Data Download](#data-download)
- [CT Data Loading and Visualization](#ct-data-loading-and-visualization)
- [Preprocessing with MONAI](#preprocessing-with-monai)
- [Running the Segmentation Model](#running-the-segmentation-model)
- [Postprocessing and Analysis](#postprocessing-and-analysis)
- [Computing Organ Volumes](#computing-organ-volumes)
- [References](#references)

---

## Prerequisites

- **Python 3.7+**
- **Packages:**  
  - `torch`  
  - `monai`  
  - `pydicom`  
  - `matplotlib`  
  - `tcia_utils`  
  - `rt-utils`  
  - `scipy`
  
You can install most of these via `pip`:
```bash
pip install torch monai pydicom matplotlib tcia_utils rt-utils scipy
Note:

Ensure you have a working GPU with CUDA support if you plan on using the model for large volumes efficiently.
This repository uses MONAI, a PyTorch-based framework specialized for medical imaging AI.
The TCIA dataset is publicly available but may require acceptance of certain usage terms.
Data Download
The code uses tcia_utils to interact with TCIA. A cart name (or Series Instance UID) is required. In the example provided:

python
Copy code
cart_name = "nbia-56561691129779503"
cart_data = nbia.getSharedCart(cart_name)
df = nbia.downloadSeries(cart_data, format="df", path=datadir)
This downloads the specified series into the designated datadir.
The datadir is set here:

python
Copy code
datadir = '/Users/sovitnayak/Documents/organ_segmentation'
Make sure to update datadir to a suitable directory on your system.

CT Data Loading and Visualization
We demonstrate two approaches to loading and viewing the CT data:

Using pydicom:

python
Copy code
ds = pydicom.read_file(os.path.join(CT_folder, '1-394.dcm'))
image = ds.pixel_array
image = ds.RescaleSlope * image + ds.RescaleIntercept
Simple 2D plotting with matplotlib can be performed after loading.

Using monai.transforms.LoadImage:

python
Copy code
from monai.transforms import LoadImage
image_loader = LoadImage(image_only=True)
CT = image_loader(CT_folder)
Using MONAI is generally simpler for 3D volumetric data. The CT data is stored with metadata including affine transformations and spacing information.

Preprocessing with MONAI
Medical imaging often requires complex preprocessing. MONAI provides a composable transform pipeline. For example:

python
Copy code
from monai.transforms import (
    EnsureChannelFirst,
    Orientation,
    Compose
)

preprocessing_pipeline = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    Orientation(axcodes='LPS')
])

CT = preprocessing_pipeline(CT_folder)
This ensures the data is in a known orientation and channel-first format, suitable for deep learning models.

Dictionary-based transforms are also supported (e.g., LoadImaged, EnsureChannelFirstd, etc.), which operate on dictionary inputs for more flexible pipelines.

Running the Segmentation Model
This repository uses a pre-trained model available via the MONAI model zoo. We download the model and its configuration:

python
Copy code
from monai.bundle import ConfigParser, download

model_name = "wholeBody_ct_segmentation"
download(name=model_name, bundle_dir=datadir)

model_path = os.path.join(datadir, model_name, 'models', 'model_lowres.pt')
config_path = os.path.join(datadir, model_name, 'configs', 'inference.json')
Load and parse the configuration:

python
Copy code
config = ConfigParser()
config.read_config(config_path)

preprocessing = config.get_parsed_content("preprocessing")
model = config.get_parsed_content("network")
inferer = config.get_parsed_content("inferer")
postprocessing = config.get_parsed_content("postprocessing")
Initialize the model with the downloaded weights:

python
Copy code
model.load_state_dict(torch.load(model_path))
model.eval()
Preprocess your data and run inference:

python
Copy code
data = preprocessing({'image': CT_folder})

with torch.no_grad():
    # Add batch dimension before inference
    data['pred'] = inferer(data['image'].unsqueeze(0), network=model)

# Remove batch dimension
data['pred'] = data['pred'][0]
data['image'] = data['image'][0]

# Apply postprocessing
data = postprocessing(data)
This yields a segmentation mask aligned with the original CT data.

Postprocessing and Analysis
The postprocessing pipeline (defined in the inference.json) may include various steps, such as saving the result as a NIfTI file. After postprocessing, you can visualize results:

python
Copy code
import matplotlib.pyplot as plt
segmentation = data['pred'][0].cpu().numpy()

slice_idx = 250
CT_slice = data['image'][0,:,slice_idx].cpu().numpy()
mask_slice = segmentation[:,slice_idx]

plt.figure(figsize=(6,8))
plt.subplot(1,2,1)
plt.imshow(CT_slice.T, cmap='Greys_r')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(mask_slice.T, cmap='nipy_spectral')
plt.axis('off')
plt.show()
Computing Organ Volumes
Once you have a segmentation, you can compute organ-specific volumes. For example, suppose the bladder is labeled as 13 in the segmentation:

python
Copy code
import numpy as np

bladder_voxels = (segmentation == 13).sum().item()
voxel_volume_cm3 = np.prod(CT.meta['spacing'] / 10)  # mm to cm
bladder_volume = bladder_voxels * voxel_volume_cm3
print(f'Bladder Volume: {bladder_volume:.1f} cm^3')
You can repeat this computation for any labeled organ in the segmentation map to quantify organ volumes.

References
MONAI
The Cancer Imaging Archive (TCIA)
Whole-body CT segmentation model on the MONAI Model Zoo
License
This code is provided under the MIT License. Refer to the license file for more details.

Disclaimer:
This code and model are for research and educational purposes only. Clinical use requires proper validation and regulatory approval.

Copy code





