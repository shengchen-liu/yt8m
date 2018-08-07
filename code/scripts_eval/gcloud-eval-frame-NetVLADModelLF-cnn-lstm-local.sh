# TRAIN_DIR="yt8m_train_local/all/gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe"
# TRAIN_DIR="yt8m_train_local/gatednetvladLF"
BUCKET_NAME=gs://${USER}_yt8m_train_bucket
JOB_TO_EVAL=gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe-cnn-lstm
JOB_NAME=yt8m_eval_$(date +%Y%m%d_%H%M%S)
MODEL_NAME=NetVLADModelLF_CNN_LSTM
TRAIN_DIR=$BUCKET_NAME/all/gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe-cnn-lstm 

gcloud  ml-engine local train  --verbosity=debug --package-path=youtube-8m \
                                                --module-name=youtube-8m.eval \
                                                -- --eval_data_pattern='gs://youtube8m-ml-us-east1/2/frame/validate/validate*.tfrecord' \
                                                --model=$MODEL_NAME \
								               --train_dir=$TRAIN_DIR  \
								                  --feature_names="rgb,audio" \
								                  --feature_sizes="1024,128" \
								                  --batch_size=80 \
								                  --base_learning_rate=0.0002 \
								                  --learning_rate_decay=0.8 \
								                  --netvlad_cluster_size=256 \
								                  --netvlad_hidden_size=1024 \
								                  --moe_l2=1e-6 \
								                  --iterations=300 \
								                  --netvlad_relu=False \
								                  --gating=True \
								                  --moe_prob_gating=True \
								                  --num_readers=4 \
								                  --sample_random_frames=False \
								                  --lstm_random_sequence=False \
								                  --multiscale_cnn_lstm_layers=4 \
								                  --moe_num_mixtures=4 \
								                  --run_once=True \
								                  --is_training=False

    






