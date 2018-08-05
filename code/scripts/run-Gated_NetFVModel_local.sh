# TRAIN_DIR="yt8m_train_local/all/gatednetfvLF-128k-1024-80-0002-300iter-norelu-basic-gatedmoe" 
TRAIN_DIR="/media/shengchen/Shengchen/yt8m/yt8m_train_local/atednetfvLF-128k-1024-80-0002-300iter-norelu-basic-gatedmoe"
# TRAIN_DIR='gs://shengchen_yt8m_sample_bucket_1/yt8m/v2/frame/train*.tfrecord'

INPUT="/media/shengchen/Shengchen/yt8m/input/frame/train*.tfrecord"
MODEL_NAME=NetFVModelLF
gcloud --verbosity=debug ml-engine local train --package-path=youtube-8m \
                                               --module-name=youtube-8m.train \
                                               -- --train_data_pattern=$INPUT \
                                               --frame_features=True \
                                               --model=$MODEL_NAME \
                                               --train_dir=$TRAIN_DIR  \
                                               --feature_names="rgb,audio" \
                                               --feature_sizes="1024,128" \
                                               --batch_size=80 \
                                               --base_learning_rate=0.0002 \
                                                --learning_rate_decay=0.8 \
                                                --fv_cluster_size=128 \
                                                --fv_hidden_size=1024 \
                                                --moe_l2=1e-6 \
                                                --iterations=300 \
                                                --fv_relu=False \
                                                --gating=True \
                                                --moe_prob_gating=True \
                                                --fv_couple_weights=False \
                                                --num_readers=4 \
                                                --num_epochs=5 

