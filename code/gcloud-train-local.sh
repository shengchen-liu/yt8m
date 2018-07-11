JOB_DIR="yt8m_train_local"
GCS_DATA_FILE="./data/data.csv"

# gcloud ml-engine local train --package-path trainer \
#                              --module-name trainer.task \
#                              -- \
#                              --data-file $GCS_DATA_FILE \
#                              --job-dir $JOB_DIR \

gcloud ml-engine local train --package-path=youtube-8m \
                             --module-name=youtube-8m.train \
                             -- --train_data_pattern='gs://youtube8m-ml-us-east1/2/video/train/train*.tfrecord' \
                             --train_dir=$JOB_DIR \
                             --model=LogisticModel \
                             --start_new_model \
                             --verbosity=debug
