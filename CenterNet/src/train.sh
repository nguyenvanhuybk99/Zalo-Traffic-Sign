python transfer_learning.py ctdet --data_dir /data \
                                  --lr 2.5e-4 --lr_step 50,80 --num_epochs 100  \
                                  --arch hourglass \
                                  --dataset zalo_challenge --exp_id zalo_challenge \
                                  --save_dir ../exp/ctdet/reproducer/ \
                                  --batch_size 8 --num_workers 0\
                                  --keep_res
