# TRAIN_DIR="yt8m_train_local/all/gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe"
# TRAIN_DIR="yt8m_train_local/gatednetvladLF"
TRAIN_DIR="gs://${USER}_yt8m_train_bucket/all"
BUCKET_NAME=gs://${USER}_yt8m_train_bucket
JOB_TO_INFER=gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe
JOB_NAME=yt8m_inference_$(date +%Y%m%d_%H%M%S)
MODEL_NAME=NetVLADModelLF

gcloud  ml-engine local train  --verbosity=debug --package-path=youtube-8m \
                                                --module-name=youtube-8m.inference \
                                                -- --input_data_pattern='gs://youtube8m-ml/2/frame/test/test*.tfrecord' \
                                                --train_dir=$TRAIN_DIR/${JOB_TO_INFER} \
                                                --batch_size=1024 \
                                                --num_readers=8 \
                                                --output_file=$BUCKET_NAME/${JOB_TO_EVAL}/predictions_gatedlightvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe.csv
                                                

    




# gcloud --verbosity=debug ml-engine jobs \
#        submit training $JOB_NAME \
#               --package-path=youtube-8m \
#               --module-name=youtube-8m.inference \
#               --staging-bucket=$BUCKET_NAME \
#               --region=us-east1 \
#               --config=youtube-8m/cloudml-gpu.yaml \
#               -- --input_data_pattern='gs://youtube8m-ml/2/frame/test/test*.tfrecord' \
#               --train_dir=$TRAIN_DIR/${JOB_TO_EVAL} \
#               --num_readers=8 \
#               --output_file=$BUCKET_NAME/${JOB_TO_EVAL}/predictions_gatedlightvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe.csv

# gsutil cp gs://shengchen_yt8m_train_bucket/all/gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe/model.ckpt-236713.data-00000-of-00001 /Users/shengchen/Documents/DataScience/yt8m/code/yt8m_train_local/all/gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe/
