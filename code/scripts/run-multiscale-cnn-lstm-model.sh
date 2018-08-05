BUCKET_NAME=gs://${USER}_yt8m_train_bucket
# BUCKET_NAME=gs://${USER}_yt8m_sample_bucket

# (One Time) Create a storage bucket to store training logs and checkpoints.
# gsutil mb -l us-east1 $BUCKET_NAME
# Submit the training job.
JOB_NAME=yt8m_train_$(date +%Y%m%d_%H%M%S)
MODEL_NAME=MultiscaleCnnLstmModel
TRAIN_DIR=$BUCKET_NAME/all/multiscale_cnn_lstm_model_batch128_decay08

gcloud --verbosity=debug ml-engine jobs submit training $JOB_NAME \
                                                      --package-path=youtube-8m \
                                                      --module-name=youtube-8m.train \
                                                      --staging-bucket=$BUCKET_NAME --region=us-central1 \
                                                      --config=youtube-8m/cloudml-4gpu.yaml \
                                                      -- --train_data_pattern='gs://youtube8m-ml-us-east1/2/frame/train/train*.tfrecord' \
                                                      --frame_features=True \
                                                      --model=$MODEL_NAME \
                                                      --feature_names="rgb,audio" \
                                                      --feature_sizes="1024,128" \
                                                      --multiscale_cnn_lstm_layers=4 \
                                                      --moe_num_mixtures=4 \
                                                      --multitask=True \
                                                      --label_loss=MultiTaskCrossEntropyLoss \
                                                      --support_loss_percent=1.0 \
                                                      --support_type="label,label,label,label" \
                                                      --is_training=True \
                                                      --batch_size=128 \
                                                      --base_learning_rate=0.001 \
                                                      --learning_rate_decay=0.8 \
                                                      --train_dir=$TRAIN_DIR  \
                                                      --num_readers=4 \
                                                      --num_epochs=5 \
                                                      # --start_new_model=True


