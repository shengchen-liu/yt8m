BUCKET_NAME=gs://${USER}_yt8m_train_bucket

JOB_TO_INFER=multiscale_cnn_lstm_model_batch128_decay08
JOB_NAME=yt8m_inference_$(date +%Y%m%d_%H%M%S)
gcloud --verbosity=debug ml-engine jobs \
       submit training $JOB_NAME \
              --package-path=youtube-8m \
              --module-name=youtube-8m.inference \
              --staging-bucket=$BUCKET_NAME \
              --region=us-central1 \
              --config=youtube-8m/cloudml-4gpu.yaml \
              -- --input_data_pattern='gs://youtube8m-ml/2/frame/test/test*.tfrecord' \
              --train_dir=$BUCKET_NAME/all/${JOB_TO_INFER} \
              --batch_size=512 \
              --num_readers=8 \
              --output_file=$BUCKET_NAME/all/${JOB_TO_EVAL}/prediction_multiscale_cnn_lstm_model_batch128_decay08.csv