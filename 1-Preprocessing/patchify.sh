# !/bin/bash
# 1. Segmentation
python3 create_patches_fp.py --source /mnt/zhen_chen/AIEC_tiff/MMRd/ --save_dir /mnt/zhen_chen/patchesx20_256/MMRd --patch_size 256 --seg --patch --stitch --no_auto_skip
python3 create_patches_fp.py --source /mnt/zhen_chen/AIEC_tiff/NSMP/ --save_dir /mnt/zhen_chen/patchesx20_256/NSMP --patch_size 256 --seg --patch --stitch --no_auto_skip
python3 create_patches_fp.py --source /mnt/zhen_chen/AIEC_tiff/P53abn/ --save_dir /mnt/zhen_chen/patchesx20_256/P53abn --patch_size 256 --seg --patch --stitch --no_auto_skip
python3 create_patches_fp.py --source /mnt/zhen_chen/AIEC_tiff/POLEmut/ --save_dir /mnt/zhen_chen/patchesx20_256/POLEmut --patch_size 256 --seg --patch --stitch --no_auto_skip

python3 extract_features_fp.py --data_h5_dir /mnt/zhen_chen/patchesx20_256/MMRd --data_slide_dir /mnt/zhen_chen/AIEC_tiff/MMRd --csv_path /mnt/zhen_chen/patchesx20_256/MMRd/process_list_autogen.csv --feat_dir /mnt/zhen_chen/featuresx20_256/MMRd --batch_size 512 --slide_ext .tif
python3 extract_features_fp.py --data_h5_dir /mnt/zhen_chen/patchesx20_256/NSMP --data_slide_dir /mnt/zhen_chen/AIEC_tiff/NSMP --csv_path /mnt/zhen_chen/patchesx20_256/NSMP/process_list_autogen.csv --feat_dir /mnt/zhen_chen/featuresx20_256/NSMP --batch_size 512 --slide_ext .tif
python3 extract_features_fp.py --data_h5_dir /mnt/zhen_chen/patchesx20_256/P53abn --data_slide_dir /mnt/zhen_chen/AIEC_tiff/P53abn --csv_path /mnt/zhen_chen/patchesx20_256/P53abn/process_list_autogen.csv --feat_dir /mnt/zhen_chen/featuresx20_256/P53abn --batch_size 512 --slide_ext .tif
python3 extract_features_fp.py --data_h5_dir /mnt/zhen_chen/patchesx20_256/POLEmut --data_slide_dir /mnt/zhen_chen/AIEC_tiff/POLEmut --csv_path /mnt/zhen_chen/patchesx20_256/POLEmut/process_list_autogen.csv --feat_dir /mnt/zhen_chen/featuresx20_256/POLEmut --batch_size 512 --slide_ext .tif