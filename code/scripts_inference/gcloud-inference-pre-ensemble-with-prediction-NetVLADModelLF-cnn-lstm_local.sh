# TRAIN_DIR="yt8m_train_local/all/gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe"
# TRAIN_DIR="yt8m_train_local/gatednetvladLF"
TRAIN_DIR="gs://${USER}_yt8m_train_bucket/all"

BUCKET_NAME=gs://${USER}_yt8m_train_bucket
JOB_TO_INFER=gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe-cnn-lstm
JOB_NAME=yt8m_inference_$(date +%Y%m%d_%H%M%S)
MODEL_NAME=MultiscaleCnnLstmModel
OUTPUT_DIR="gs://${USER}_yt8m_train_bucket/all/model_predictions/${JOB_TO_INFER}"
OUTPUT_FILE=$OUTPUT_DIR/${JOB_TO_INFER}_predictions.csv

gcloud  ml-engine local train  --verbosity=debug --package-path=youtube-8m \
                                                --module-name=youtube-8m.inference \
                                                -- --input_data_pattern='gs://youtube8m-ml/2/frame/test/test*.tfrecord' \
                                                -output_dir=$OUTPUT_DIR \
                                                  --output_file=$OUTPUT_FILE \
                                                  --train_dir=$TRAIN_DIR/${JOB_TO_INFER} \
                                                  --model=$MODEL_NAME \
                                                   --frame_features=True \
                                                   --feature_names="rgb,audio" \
                                                   --feature_sizes="1024,128" \
                                                  --num_readers=8 \
                                                  --batch_size=128 \
                                                   --file_size=4096 \
                                                  --base_learning_rate=0.0002 \
                                                  --netvlad_cluster_size=256 \
                                                  --netvlad_hidden_size=1024 \
                                                  --moe_l2=1e-6 \
                                                  --iterations=300 \
                                                  --learning_rate_decay=0.8 \
                                                  --netvlad_relu=False \
                                                  --gating=True \
                                                  --moe_prob_gating=False \
                                                  --lightvlad=False \
                                                  --run_once=True \
                                                   --moe_num_mixtures=4 \
                                                  --is_training=False                        
                                                # --output_file=$BUCKET_NAME/${JOB_TO_EVAL}/predictions_multiscale_cnn_lstm_model_batch128_decay08.csv
                                                

    




BUCKET_NAME=gs://${USER}_yt8m_train_bucket
TRAIN_DIR=gs://${USER}_yt8m_train_bucket/all
JOB_TO_INFER=gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe-cnn-lstm
OUTPUT_DIR="gs://${USER}_yt8m_train_bucket/all/model_predictions/${JOB_TO_INFER}"
OUTPUT_FILE=$OUTPUT_DIR/${JOB_TO_INFER}_predictions.csv
JOB_NAME=yt8m_inference_$(date +%Y%m%d_%H%M%S)
MODEL_NAME=NetVLADModelLF_CNN_LSTM

gcloud --verbosity=debug ml-engine jobs \
       submit training $JOB_NAME \
              --package-path=youtube-8m \
              --module-name=youtube-8m.inference\
              --staging-bucket=$BUCKET_NAME \
              --region=us-central1 \
              --config=youtube-8m/cloudml-gpup100.yaml \
              -- --input_data_pattern='gs://youtube8m-ml/2/frame/test/test*.tfrecord' \
              --output_dir=$OUTPUT_DIR \
              --output_file=$OUTPUT_FILE \
              --train_dir=$TRAIN_DIR/${JOB_TO_INFER} \
              --model=$MODEL_NAME \
               --frame_features=True \
               --feature_names="rgb,audio" \
               --feature_sizes="1024,128" \
              --num_readers=8 \
              --batch_size=128 \
               --file_size=4096 \
              --base_learning_rate=0.0002 \
              --netvlad_cluster_size=256 \
              --netvlad_hidden_size=1024 \
              --moe_l2=1e-6 \
              --iterations=300 \
              --learning_rate_decay=0.8 \
              --netvlad_relu=False \
              --gating=True \
              --moe_prob_gating=False \
              --lightvlad=False \
              --run_once=True \
               --moe_num_mixtures=4 \
              --is_training=False