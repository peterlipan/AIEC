# basedownsample=2 -> x20 magnification
python3 create_patches.py --src /mnt/zhen_chen/AIEC_tiff/MMRd \
    --dst /mnt/zhen_chen/patches_TongJi_DTFD_x10_down_2_2/MMRd \
    --patch_size 256 --base_downsample 4 --num_levels 3 --downsample_factor "2,2"

python3 create_patches.py --src /mnt/zhen_chen/AIEC_tiff/NSMP \
    --dst /mnt/zhen_chen/patches_TongJi_DTFD_x10_down_2_2/NSMP \
    --patch_size 256 --base_downsample 4 --num_levels 3 --downsample_factor "2,2"

python3 create_patches.py --src /mnt/zhen_chen/AIEC_tiff/P53abn \
    --dst /mnt/zhen_chen/patches_TongJi_DTFD_x10_down_2_2/P53abn \
    --patch_size 256 --base_downsample 4 --num_levels 3 --downsample_factor "2,2"
    
python3 create_patches.py --src /mnt/zhen_chen/AIEC_tiff/POLEmut \
    --dst /mnt/zhen_chen/patches_TongJi_DTFD_x10_down_2_2/POLEmut \
    --patch_size 256 --base_downsample 4 --num_levels 3 --downsample_factor "2,2"