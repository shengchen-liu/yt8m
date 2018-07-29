# TRAIN_DIR="yt8m_train_local/all/gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe"
# TRAIN_DIR="yt8m_train_local/gatednetvladLF"
TRAIN_DIR="gs://${USER}_yt8m_train_bucket/all"
BUCKET_NAME=gs://${USER}_yt8m_train_bucket
JOB_TO_EVAL=gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe
JOB_NAME=yt8m_eval_$(date +%Y%m%d_%H%M%S)
MODEL_NAME=NetVLADModelLF

gcloud  ml-engine local train  --verbosity=debug --package-path=youtube-8m \
                                                --module-name=youtube-8m.eval \
                                                -- --eval_data_pattern='gs://youtube8m-ml-us-east1/2/frame/validate/validate*.tfrecord' \
                                                --model=$MODEL_NAME \
                                                --train_dir=$TRAIN_DIR/${JOB_TO_EVAL} \
                                                --run_once=True \
                                                --frame_features=True \
                                                --feature_names='rgb,audio' \
                                                --feature_sizes="1024,128" \
                                                --batch_size=1024 \
                                                --netvlad_cluster_size=256 \
                                                --netvlad_hidden_size=1024 \
                                                --moe_l2=1e-6 \
                                                --iterations=300 \
                                                --learning_rate_decay=0.8 \
                                                --netvlad_relu=False  \
                                                --gating=True \
                                                --moe_prob_gating=False \
                                                --is_training=False \
                                                --moe_num_mixtures=2

    







# MODEL_NAME=NetVLADModelLF
# gcloud ml-engine local train --package-path=youtube-8m \
#                              --module-name=youtube-8m.eval \
#                              -- --eval_data_pattern='gs://youtube8m-ml-us-east1/2/frame/validate/validate*.tfrecord' \
#                              --frame_features=True \
#                              --model=$MODEL_NAME \
#                              --feature_names="rgb,audio" \
#                              --feature_sizes="1024,128" \
#                              --multiscale_cnn_lstm_layers=4 \
#                              --moe_num_mixtures=4 \
#                              --multitask=True \
#                              --label_loss=MultiTaskCrossEntropyLoss \
#                              --support_loss_percent=1.0 \
#                              --support_type="label,label,label,label" \
#                              --is_training=True \
#                              --batch_size=64 \
#                              --base_learning_rate=0.001 \
#                              --train_dir=$TRAIN_DIR  \
#                              --num_readers=4 \
#                              --num_epochs=5 \
#                              --start_new_model=True \
#                              --verbosity=debug