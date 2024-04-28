# !/bin/bash
# 1. Segmentation
python3 create_patches_fp.py --source /mnt/zhen_chen/AIEC_tiff/MMRd/ --save_dir /mnt/zhen_chen/patchesx20_256/MMRd --patch_size 256 --seg --patch --stitch --no_auto_skip
python3 create_patches_fp.py --source /mnt/zhen_chen/AIEC_tiff/NSMP/ --save_dir /mnt/zhen_chen/patchesx20_256/NSMP --patch_size 256 --seg --patch --stitch --no_auto_skip
python3 create_patches_fp.py --source /mnt/zhen_chen/AIEC_tiff/P53abn/ --save_dir /mnt/zhen_chen/patchesx20_256/P53abn --patch_size 256 --seg --patch --stitch --no_auto_skip
python3 create_patches_fp.py --source /mnt/zhen_chen/AIEC_tiff/POLEmut/ --save_dir /mnt/zhen_chen/patchesx20_256/POLEmut --patch_size 256 --seg --patch --stitch --no_auto_skip