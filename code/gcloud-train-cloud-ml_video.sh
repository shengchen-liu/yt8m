
# JOB_NAME="movie_sentiment_"$(date +%s)
# JOB_DIR="gs://lawson_movie/sentiment_keras_cloud_ml"
# GCS_DATA_FILE="gs://lawson_movie/data.csv"

# gcloud ml-engine jobs submit training $JOB_NAME \
#                                     --stream-logs \
#                                     --runtime-version 1.8 \
#                                     --job-dir $JOB_DIR \
#                                     --package-path trainer \
#                                     --module-name trainer.task \
#                                     --region us-east1 \
#                                     -- \
#                                     --data-file $GCS_DATA_FILE 
#                                     --verbosity=debug


BUCKET_NAME=gs://${USER}_yt8m_train_bucket
# (One Time) Create a storage bucket to store training logs and checkpoints.
# gsutil mb -l us-east1 $BUCKET_NAME
# Submit the training job.
JOB_NAME=yt8m_train_$(date +%Y%m%d_%H%M%S)
gcloud --verbosity=debug ml-engine jobs submit training $JOB_NAME \
                                                      --package-path=youtube-8m \
                                                      --module-name=youtube-8m.train \
                                                      --staging-bucket=$BUCKET_NAME --region=us-east1 \
                                                      --config=youtube-8m/cloudml-gpu.yaml \
                                                      -- --train_data_pattern='gs://youtube8m-ml-us-east1/2/video/train/train*.tfrecord' \
                                                      --model=LogisticModel \
                                                      --train_dir=$BUCKET_NAME/yt8m_train_video_level_logistic_model
