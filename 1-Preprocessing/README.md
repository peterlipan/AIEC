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



```sh
curl -H "Authorization: Bearer ya29.a0Ad52N3-Ngj4C0-ShnSl9mcueyxVGukKWxxvueQw5qLxjLXcGW5ASWXYWHmg5sf92ugTvwU84zx-2KMIXb0t96E9Nxdc3RuPYm0UJ3LPRDN75CAQs8tpgbfiuPqIeR_4842y2si4w21BOEYGCekZ_ug9SZd0VZeO35MOUaCgYKAUUSARISFQHGX2Mimbct1yE0l_mNVazGJWJGfw0171" -C - "https://www.googleapis.com/drive/v3/files/1hE8FHKcL3-qYJXM9JPvyrS76cI69fcrm?alt=media" -o AIEC_tiff

wget --recursive --no-parent --header="Authorization: Bearer ya29.a0Ad52N3_OO3-_vreqLa6ZN0gBO6E8RabXig2abZbMXLr-vv4cpIjUtNgVpp49rwbEpfllPPJynFswJcPYeAX20mHd62-TmR5zmMsvwjQ6q3Nps1s_wCy2zhz6lFcc10Sz82EvGHZYr-TdgyoWpw28vjfs2V3yx79ckIsqaCgYKAcISARISFQHGX2Mi8P6IPfOmVsxkBtCOk-M36g0171" "https://www.googleapis.com/drive/v3/files/1hE8FHKcL3-qYJXM9JPvyrS76cI69fcrm?alt=media"

bypy --processes 16 --downloader aria2 downdir AIEC_202404/AIEC_tiff ./
```





