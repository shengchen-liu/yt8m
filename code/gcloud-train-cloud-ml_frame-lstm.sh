BUCKET_NAME=gs://${USER}_yt8m_train_bucket
# BUCKET_NAME=gs://${USER}_yt8m_sample_bucket

# (One Time) Create a storage bucket to store training logs and checkpoints.
# gsutil mb -l us-east1 $BUCKET_NAME
# Submit the training job.
JOB_NAME=yt8m_train_$(date +%Y%m%d_%H%M%S)
MODEL_NAME=LstmModel
TRAIN_DIR=$BUCKET_NAME/sample/lsmt_rgb_audio

gcloud --verbosity=debug ml-engine jobs submit training $JOB_NAME \
                                                      --package-path=youtube-8m \
                                                      --module-name=youtube-8m.train \
                                                      --staging-bucket=$BUCKET_NAME --region=us-east1 \
                                                      --config=youtube-8m/cloudml-4gpu.yaml \
                                                      -- --train_data_pattern='gs://shengchen_yt8m_sample_bucket/2/frame/train/train*.tfrecord' \
                                                      # -- --train_data_pattern='gs://youtube8m-ml-us-east1/2/frame/train/train*.tfrecord' \
                                                      --frame_features=True \
                                                      --model=$MODEL_NAME \
                                                      --feature_names="rgb,audio" \
                                                      --feature_sizes="1024,128" \
                                                      --batch_size=128 \
                                                      --base_learning_rate=0.001 \
                                                      --train_dir=$TRAIN_DIR  \
                                                      --num_epoch=50 \
                                                      --start_new_model=True