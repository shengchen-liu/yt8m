BUCKET_NAME=gs://${USER}_yt8m_train_bucket
TRAIN_DIR="gs://${USER}_yt8m_train_bucket/all"
JOB_TO_INFER=multiscale_cnn_lstm_model_batch128_decay08
JOB_NAME=yt8m_inference_$(date +%Y%m%d_%H%M%S)
MODEL_NAME=MultiscaleCnnLstmModel
OUTPUT_DIR="gs://${USER}_yt8m_train_bucket/all/model_predictions/${JOB_TO_INFER}"

gcloud --verbosity=debug ml-engine jobs \
       submit training $JOB_NAME \
              --package-path=youtube-8m \
              --module-name=youtube-8m.inference-pre-ensemble \
              --staging-bucket=$BUCKET_NAME \
              --region=us-central1 \
              --config=youtube-8m/cloudml-4gpu.yaml \
              -- --input_data_pattern='gs://youtube8m-ml/2/frame/test/test*.tfrecord' \
              --output_dir=$OUTPUT_DIR\
             --train_dir=$TRAIN_DIR/${JOB_TO_INFER} \
             --model=$MODEL_NAME \
               --frame_features=True \
               --feature_names="rgb,audio" \
               --feature_sizes="1024,128" \
             --batch_size=512 \
             --num_readers=8 \
             --file_size=4096


