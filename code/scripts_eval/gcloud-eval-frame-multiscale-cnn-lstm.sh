BUCKET_NAME=gs://${USER}_yt8m_train_bucket
JOB_TO_EVAL=multiscale_cnn_lstm_model_batch128_decay08
JOB_NAME=yt8m_eval_$(date +%Y%m%d_%H%M%S)MultiscaleCnnLstmModel
MODEL_NAME=MultiscaleCnnLstmModel

gcloud --verbosity=debug ml-engine jobs \
       submit training $JOB_NAME \
                --package-path=youtube-8m \
                --module-name=youtube-8m.eval \
                --staging-bucket=$BUCKET_NAME \
                --region=us-central1 \
                --config=youtube-8m/cloudml-4gpu.yaml \
                -- --eval_data_pattern='gs://youtube8m-ml-us-east1/2/frame/validate/validate*.tfrecord' \
                --model=$MODEL_NAME \
                --train_dir=$BUCKET_NAME/all/${JOB_TO_EVAL} \
                --is_training=False \
                --run_once=True \
                --start_new_model=True \
