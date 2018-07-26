# CUDA_VISIBLE_DEVICES=1 python train.py \
# 	--train_dir="../model/lstmparallelfinaloutput1024_moe8" \
# 	--frame_features=True \
# 	--feature_names="rgb,audio" \
# 	--feature_sizes="1024,128" \
# 	--train_data_pattern="/Youtube-8M/data/frame/train/train*" \
# 	--batch_size=128 \
# 	--lstm_cells="1024,128" \
# 	--moe_num_mixtures=8 \
# 	--model=LstmParallelFinaloutputModel \
# 	--rnn_swap_memory=True \
# 	--num_readers=1 \
# 	--num_epochs=3 \
# 	--base_learning_rate=0.001




BUCKET_NAME=gs://${USER}_yt8m_train_bucket
# BUCKET_NAME=gs://${USER}_yt8m_sample_bucket

# (One Time) Create a storage bucket to store training logs and checkpoints.
# gsutil mb -l us-east1 $BUCKET_NAME
# Submit the training job.
JOB_NAME=yt8m_train_$(date +%Y%m%d_%H%M%S)
MODEL_NAME=LstmParallelFinaloutputModel
TRAIN_DIR=$BUCKET_NAME/all/lstmparallelfinaloutput1024_moe8

gcloud --verbosity=debug ml-engine jobs submit training $JOB_NAME \
                                                      --package-path=youtube-8m \
                                                      --module-name=youtube-8m.train \
                                                      --staging-bucket=$BUCKET_NAME --region=us-central1 \
                                                      --config=youtube-8m/cloudml-8gpu.yaml \
                                                      -- --train_data_pattern='gs://youtube8m-ml-us-east1/2/frame/train/train*.tfrecord' \
                                                      --frame_features=True \
                                                      --model=$MODEL_NAME \
                                                      --feature_names="rgb,audio" \
                                                      --feature_sizes="1024,128" \
                                                      --moe_num_mixtures=8 \
                                                      --batch_size=128 \
                                                      --base_learning_rate=0.001 \
                                                      --train_dir=$TRAIN_DIR  \
                                                      --num_epoch=3 \
                                                      --start_new_model=True
