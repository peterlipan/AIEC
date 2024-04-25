# Preprocessing of the AIEC dataset
The dataset contains the immunostaining images of 396 patients (90 MMRd, 161 NSMP, 95 P53abn, 50 POLEmut). The raw structure of this dataset is as follows:

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

Following CLAM (https://github.com/mahmoodlab/CLAM/tree/master), we first segment the tissues into patches, excluding background and holes.

天亮，我刚刚和陈老师对了下算法的进度和设计，发现对数据这边有些问题还不太清楚。

一个是现在的图片数据是免疫组化的扫描，表格数据中包括了多种分子标志物和基因突变信息，现在不清楚能不能从单张图片中看出所有的分子信息，还是说一张免疫组化只针对特定的部分分子标志物

第二个问题是病理医生在评估EC图片的时候，会不会不断地切换放大倍率。比如在低倍镜下发现了可疑区域，需要切换到高倍镜才能确定病变程度。如果有这种逻辑的话，我们可以把金字塔结构的信息加入到模型设计中





