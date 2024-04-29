# Preprocessing of the AIEC dataset
## Raw data

The TIF dataset contains the PDL-1 immunostaining images of 396 patients (90 MMRd, 161 NSMP, 95 P53abn, 50 POLEmut). The raw structure of this dataset is as follows:

```
AIEC_tiff/
	├── MMRd
				├── 2010-0173-5.tif
				├── 2010-01233-9.tif
				└──	...
	├── NSMP
				├── 2010-0078-2.tif
				├── 2010-1557-2.tif
				└──	...
	├── P53abn
				├── 2010-5620-2.tif
				├── 2010-9173-1.tif
				└──	...
	├── POLEmut
				├── 2010-1247-2.tif
				├── 2010-1556-2.tif
				└──	...
	└──	EC_DAT.xlsx
```

Modalities in the raw dataset:

- Immunohistochemistry (IHC) staining images: CD3/CD8/**PD-L1**
- Biomarkers
  - Cancer tissues: CK7, Vimentin, ER, PR, p53, Ki-67, CD10, p16, CA125, others
  - Vascular: CD34 (blood vessels) , D2-40 (lymphatic vessels)
  - Gene mutation: SPOP, POLE
  - Gene expression: PMS2, MLH1, MSH2, MSH6, PD-L1, CD3, CD8
- Diagnosis
  - Pathological classification: Carcinomas/Carcinosarcomas/...
  - Reports: pathology, MRI, B-Scan, 
  - Molecular classification: MMRd, NSMP, P53abn, POLEmut
  - Grading
- Prognosis: Survival status (Survived/Died/Censored) and Death date
- Therapy: Radiation/Chemo

<u>**TODO:**</u> Transform the TIF images to pyramidal tiffs (ndpi/svs/ometiff) for patch segmentation and pyramidal modeling (https://github.com/mahmoodlab/CLAM/issues/241).

## Preprocessing

The prepprocessing consists of two steps: Patching and Feature extraction. Following CLAM (https://github.com/mahmoodlab/CLAM), we patchify the tissue region (excluding the background and holdes) into 256x256 patches at 20x magnification as follows:

```bash
python3 create_patches_fp.py --source /mnt/zhen_chen/AIEC_tiff/MMRd/ --save_dir /mnt/zhen_chen/patchesx20_256/MMRd --patch_size 256 --seg --patch --stitch --no_auto_skip
```

This will generate three folders and one csv file under the save_dir. The **masks** folder contains the segmentation results (one image per slide). The **patches** folder contains arrays of extracted tissue patches from each slide (one .h5 file per slide, where each entry corresponds to the coordinates of the top-left corner of a patch) The **stitches** folder contains downsampled visualizations of stitched tissue patches (one image per slide) (Optional, not used for downstream tasks) The auto-generated csv file **process_list_autogen.csv** contains a list of all slides processed, along with their segmentation/patching parameters used.



Pre-trained neural networks have been used to extract the morphological features from the segmented patches. An example script is as follows:

```bash
python3 extract_features_fp.py --data_h5_dir /mnt/zhen_chen/patchesx20_256/MMRd --data_slide_dir /mnt/zhen_chen/AIEC_tiff/MMRd --csv_path /mnt/zhen_chen/patchesx20_256/MMRd/process_list_autogen.csv --feat_dir /mnt/zhen_chen/featuresx20_256/MMRd --batch_size 512 --slide_ext .tif
```

The above command expects the coordinates .h5 files to be stored under data_h5_dir and a batch size of 512 to extract **1024-dim features** from each tissue patch for each slide and produce tow folders: **h5_files** and **pt_files**. Each .h5 file contains an array of extracted features along with their patch coordinates (note for faster training, a .pt file for each slide is also created for each slide, containing just the patch features). Current version utilized a ResNet-50 pretrained on ImageNet for this step. The final folder structure:

```
featuresx20_256/
	├── MMRd
				├── h5_files # patch-level feautres
					└── ...
				└── pt_files # patch-level feautres for faster training
					└──	...
	├── NSMP
				├── h5_files
					└── ...
				└── pt_files
					└──	...
	├── P53abn
				├── h5_files
					└── ...
				└── pt_files
					└──	...
	└── POLEmut
				├── h5_files
					└── ...
				└── pt_files
					└──	...
```

**<u>TODO:</u>** Implement UNI and CONCH (ViTs pre-trained on pathology slides) for feature extraction.

