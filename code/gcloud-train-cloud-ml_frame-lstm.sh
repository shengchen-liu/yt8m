BUCKET_NAME=gs://${USER}_yt8m_train_bucket
# (One Time) Create a storage bucket to store training logs and checkpoints.
# gsutil mb -l us-east1 $BUCKET_NAME
# Submit the training job.
JOB_NAME=yt8m_train_$(date +%Y%m%d_%H%M%S)
gcloud --verbosity=debug ml-engine jobs submit training $JOB_NAME \
                                                      --package-path=youtube-8m \
                                                      --module-name=youtube-8m.train \
                                                      --staging-bucket=$BUCKET_NAME --region=us-east1 \
                                                      --config=youtube-8m/cloudml-4gpu.yaml \
                                                      -- --train_data_pattern='gs://youtube8m-ml-us-east1/2/frame/train/train*.tfrecord' \
                                                      --frame_features=True \
                                                      --model=LstmModel \
                                                      --feature_names="rgb" \
                                                      --feature_sizes="1024" \
                                                      --batch_size=128 \
                                                      --base_learning_rate=0.001 \
                                                      --train_dir=$BUCKET_NAME/yt8m_train_lstm_model/model_epoch5 \
                                                      --start_new_model