BUCKET_NAME=gs://${USER}_yt8m_train_bucket
TRAIN_DIR=gs://${USER}_yt8m_train_bucket/all
JOB_TO_INFER=gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe
OUTPUT_DIR="gs://${USER}_yt8m_train_bucket/all/model_predictions/${JOB_TO_INFER}"
JOB_NAME=yt8m_inference_$(date +%Y%m%d_%H%M%S)
MODEL_NAME=NetVLADModelLF

gcloud --verbosity=debug ml-engine jobs \
       submit training $JOB_NAME \
              --package-path=youtube-8m \
              --module-name=youtube-8m.inference-pre-ensemble \
              --staging-bucket=$BUCKET_NAME \
              --region=us-central1 \
              --config=youtube-8m/cloudml-gpu.yaml \
              -- --input_data_pattern='gs://youtube8m-ml/2/frame/test/test*.tfrecord' \
              --output_dir=$OUTPUT_DIR\
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
              --moe_num_mixtures=2 \
              --is_training=False


