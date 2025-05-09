# python3 main.py --weight_decay 0.0001
# python3 main.py --weight_decay 0.001
# python3 main.py --weight_decay 0.01
# python3 main.py --weight_decay 0.1

# Best
# python3 main.py --weight_decay 1.0 --lr 2e-4 --beta1 0.9 --beta2 0.99 --epochs 400 --warmup_epochs 20 --d_state 32 --batch_size 3 --tree_dropout "0.08, 0.04, 0.01"

# python3 main.py --weight_decay 1.0 --lr 2e-4 --beta1 0.9 --beta2 0.99 --epochs 400 \
# --warmup_epochs 20 --d_state 32 --batch_size 3 --tree_dropout "0.08, 0.04, 0.01" \
# --data_root /mnt/zhen_chen/features_TongJi_DTFD_x20_down_2_4 --downsample_factor "2, 4" \
# --train_csv ./TongJi_AIEC_Training_Aug.csv --fix_agent 1 --random_layer 0.5

# with xview

# python3 main.py --weight_decay 1 --lr 2e-4 --beta1 0.9 --beta2 0.99 --epochs 400 --depth 1 --num_layers 1 \
# --warmup_epochs 20 --d_state 32 --batch_size 7 --tree_dropout "0.1, 0.05, 0.01" \
# --data_root /mnt/zhen_chen/features_TongJi_DTFD_x20_down_2_4 --downsample_factor "2, 4" \
# --train_csv TongJi_AIEC_Training_Final.csv --test_csv TongJi_AIEC_Testing_Final.csv \
# --fix_agent 1 --random_layer 0.1 --lambda_xview 0.05 --seed 2

python3 main.py --weight_decay 1 --lr 2e-4 --epochs 200 --n_layers 1
python3 main.py --weight_decay 1 --lr 2e-4 --epochs 200 --n_layers 2

python3 main.py --weight_decay 1 --lr 2e-4 --epochs 200 --n_layers 1 --n_views 5
python3 main.py --weight_decay 1 --lr 2e-4 --epochs 200 --n_layers 2 --n_views 5

python3 main.py --weight_decay 1 --lr 2e-4 --epochs 200 --n_layers 1 --n_views 6
python3 main.py --weight_decay 1 --lr 2e-4 --epochs 200 --n_layers 2 --n_views 6

python3 main.py --weight_decay 1 --lr 2e-4 --epochs 200 --n_layers 1 --n_views 7
python3 main.py --weight_decay 1 --lr 2e-4 --epochs 200 --n_layers 2 --n_views 7
python3 main.py --weight_decay 1 --lr 2e-4 --epochs 200 --n_layers 1 --n_views 8
python3 main.py --weight_decay 1 --lr 2e-4 --epochs 200 --n_layers 2 --n_views 8