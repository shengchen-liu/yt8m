BUCKET_NAME=gs://${USER}_yt8m_train_bucket

JOB_TO_INFER=LstmParallelFinaloutputModel
JOB_NAME=yt8m_inference_$(date +%Y%m%d_%H%M%S)
gcloud --verbosity=debug ml-engine jobs \
       submit training $JOB_NAME \
              --package-path=youtube-8m \
              --module-name=youtube-8m.inference \
              --staging-bucket=$BUCKET_NAME \
              --region=us-central1 \
              --config=youtube-8m/cloudml-gpu.yaml \
              -- --input_data_pattern='gs://youtube8m-ml/2/frame/test/test*.tfrecord' \
              --train_dir=$BUCKET_NAME/all/${JOB_TO_INFER} \
              --num_readers=8 \
              --output_file=$BUCKET_NAME/all/${JOB_TO_INFER}/predictions.csv