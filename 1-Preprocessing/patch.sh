# basedownsample=2 -> x20 magnification
python3 create_patches.py --src /mnt/zhen_chen/AIEC_tiff/MMRd \
    --dst /mnt/zhen_chen/patches_TongJi_DTFD_x20_down_2_4/MMRd \
    --patch_size 256 --base_downsample 2 --num_levels 3 --downsample_factor "2,4"
python3 create_patches.py --src /mnt/zhen_chen/AIEC_tiff/NSMP \
    --dst /mnt/zhen_chen/patches_TongJi_DTFD_x20_down_2_4/NSMP \
    --patch_size 256 --base_downsample 2 --num_levels 3 --downsample_factor "2,4"
python3 create_patches.py --src /mnt/zhen_chen/AIEC_tiff/P53abn \
    --dst /mnt/zhen_chen/patches_TongJi_DTFD_x20_down_2_4/P53abn \
    --patch_size 256 --base_downsample 2 --num_levels 3 --downsample_factor "2,4"
python3 create_patches.py --src /mnt/zhen_chen/AIEC_tiff/POLEmut \
    --dst /mnt/zhen_chen/patches_TongJi_DTFD_x20_down_2_4/POLEmut \
    --patch_size 256 --base_downsample 2 --num_levels 3 --downsample_factor "2,4"