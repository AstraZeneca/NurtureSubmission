# Virtual staining experiments: NURTuRE

## Software requirements
Linux platforms are supported - as long as the dependencies are supported on these platforms.

Anaconda or Miniconda with Python >= 3.10

The software has been developed on a Linux platform.

Python libraries and their versions are in [requirements.conda.yaml](requirements.conda.yaml).

The exact requirements for training are in [exact-training-requirements.conda.yaml](exact-training-requirements.conda.yaml).


## Building the virtual Python environment
To create and activate the Python env, run
```bash
conda env create -f requirements.conda.yaml
conda activate kidney-stain
```

## Training the virtual staining model

### Preparing the data
To train the virtual staining model, we used an internal patch-extraction tool which is not available. However, to reproduce the directory structure in which we save the patches, create an arbitrary directory

```bash
patch-parent-dir/
├── slide-1.svs
└── slide-2.svs
```
Note: ``slide-1.svs`` and ``slide-1.svs`` are _directories_. Inside each of these directories, the patches are saved in the following way

```bash
NURTuRE5671.svs/
├── mask.png
├── NURTuRE5671-masked.png
├── NURTuRE5671.png
├── patches
│   └── mag_20.0.zip
└── patches-mag_20.0.csv
```
The ``.png`` files in the root directory here are not essential; there are just by products of the patch extractor method. ``patches/mag_20.0.zip``—which contains the png patches used for training—and ``patches-mag_20.csv`` (coordinates and metadata for each patch) are essential.


### Running the code
Once you have prepared the data accordingly, you can train the virtual staining model by running
```bash
./scripts/train_virtual_stain.py metadata.csv /path/to/patch/parent/dir/ /save/dir/of/your/choice
```
Remember, running ``./scripts/train_virtual_stain.py --help`` is your friend.


## Training the glomerular segmentation model

### Preparing the data

Note: for each of the scripts you run here, you can always add ``--help`` for more information on optional arguments, etc. Furthermore, the patch extraction parameters, such as patch size, stride, etc., are detailed in the manuscript.

---
#### 1. KPMP
Download the PAS-stained glomerular segmentation slides from the KPMP repository, and place them in a directory like so:
```bash
/path/to/folder/KPMP
├── masks
    ├── ...
    ├── ...
    ├── ...
└── wsi
    ├── ...
    ├── ...
    ├── ...
```
At the time of this writing, you could find these data [here](https://atlas.kpmp.org/repository/?size=n_20_n&filters%5B0%5D%5Bfield%5D=workflow_type&filters%5B0%5D%5Bvalues%5D%5B0%5D=Segmentation%20Data&filters%5B0%5D%5Btype%5D=any).

Then, to extract the patches, run
```bash
./scripts/extract_kpmp_patches.py /path/to/folder/KPMP/ /path/to/save/dir/
```
---

#### 2. HuBMAP Kidney
At the time of this writing, the HuBMAP kidney data were available [here](https://www.kaggle.com/c/hubmap-kidney-segmentation). Download and unzip them so you have a file structure that looks like this:
```bash
/path/to/images/hubmap/
└── train
    ├── 0486052bb-anatomical-structure.json
    ├── 0486052bb.json
    ├── 0486052bb.tiff
```


To extract the patches, run
```bash
./scripts/extract_hubmap_patches.py /path/to/images/hubmap/ /path/to/save/dir/
```


#### 3. Jayapandian et al.

- These data were released as part of [this publication](https://www.sciencedirect.com/science/article/pii/S0085253820309625?via%3Dihub).
- At the time of this writing, these data were available to download [here](https://github.com/ccipd/DL-kidneyhistologicprimitives).



### The Imitation Game
The patches, or regions of interest, we used derive from KPMP slides. The metadata necessary to access each of these images is given in [this ``.csv`` file](imitation-game/imitation-game-img-metadata.csv). This file also contains the metadata sufficient to download the slides from the KPMP repo.




### Inference with trained models.

#### 1. Virtual staining.
To infer on a directory of patches using the virtual staining model, run:
```bash
./scripts/infer_virtual_stain.py /path/to/patch/directory/ --weights "/path/to/model/weights.pth" --out-dir "/directory/of/your/choice/"
```
Whether you use the model which goes from H&E to PAS, or _vice versa_, depends on the parameters you load.



#### 2. Glomerular segmentation.

To run inference with the glomerular segmentation model, the user must first create a directory structure like so:
```bash
test-patches/processed-patches
├── he-patches
├── masks
└── vpas-patches
```
Note: the parent directory itself is arbitrary.

Store the model weights in a directory structured like:
```bash
checkpoints/glom-seg
├── KPMP
│   └── 22.pth
├── hubmap
│   └── 22.pth
└── neptune
    └── 22.pth
```

Finally, to run inference with the trained CV models, use:
```bash
./scripts/test_segmentation_model.py test-patches/processed-patches/ /path/to/model/checkpoints/
```

## Notes

### KPMP specific

- In the KPMP dataset, the WSI and and segmentation mask in the zip file ``7b1d3fc9-8f25-4147-8ac1-ba1213611f54_segmentation_masks.zip`` have different sizes, so we remove it from all experiments. The WSI is actually a duplicate from another zip file.
- The slide ``"S-2010-004184_PAS_1of2.svs"`` seems to cause ``TiffFile`` to crash or hang, so we also exclude it.


## KPMP test set annotations
The glomerular annotations for the KPMP test set are in [this folder](data/KPMP-masks). You can open the annotations in QuPath: first download the WSIs from the KPMP repository, and then open the ``.geojson`` files in QuPath.

The metadata required to download these images from the KPMP repository is given in [this ``.csv`` file](data/KPMP-masks/kpmp-metadata.csv).

We have also included the [groovy script](data/KPMP-masks/labelled_patches.groovy), used in QuPath, for exporting labelled patches from these WSIs.



## Licence

The software is licensed under the MIT license (see LICENSE file).
