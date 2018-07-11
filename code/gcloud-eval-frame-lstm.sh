BUCKET_NAME=gs://${USER}_yt8m_train_bucket
JOB_TO_EVAL=yt8m_train_lstm_model
JOB_NAME=yt8m_eval_$(date +%Y%m%d_%H%M%S)

gcloud --verbosity=debug ml-engine jobs \
       submit training $JOB_NAME \
                --package-path=youtube-8m \ 
                --module-name=youtube-8m.eval \
                --staging-bucket=$BUCKET_NAME \
                --region=us-east1 \
                --config=youtube-8m/cloudml-4gpu.yaml \
                -- --eval_data_pattern='gs://youtube8m-ml-us-east1/2/frame/validate/validate*.tfrecord' \
                --model=LstmModel \
                --train_dir=$BUCKET_NAME/${JOB_TO_EVAL} \
                --run_once=True
