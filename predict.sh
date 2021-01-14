#cd /zalo_traffic_sign/CenterNet/src
#source ~/anaconda3/etc/profile.d/conda.sh
#conda activate base
#conda activate zalo
#sh predict.sh

cd CenterNet/src
source activate zalo
export PYTHONPATH=.
sh predict.sh
