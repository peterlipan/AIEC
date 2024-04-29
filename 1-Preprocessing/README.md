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





