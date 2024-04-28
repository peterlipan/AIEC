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

**TODO:** Transform the TIF images to pyramidal tiffs (ndpi/svs/ometiff) for patch segmentation and pyramidal modeling (https://github.com/mahmoodlab/CLAM/issues/241).

```





