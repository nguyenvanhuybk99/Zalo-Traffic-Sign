python predict.py ctdet --data_dir ../data/ \
			      --load_model ../model/model_best.pth \
			      --exp_id zalo_challenge --dataset zalo_challenge \
			      --test_scales 1,1.5  --save_dir ../exp/ctdet/ \
            --debug_huy True --keep_res
