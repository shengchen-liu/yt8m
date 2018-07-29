BUCKET_NAME=gs://${USER}_yt8m_train_bucket
TRAIN_DIR="gs://${USER}_yt8m_train_bucket/all"
JOB_TO_EVAL=gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe
JOB_NAME=yt8m_eval_$(date +%Y%m%d_%H%M%S)
MODEL_NAME=NetVLADModelLF

gcloud --verbosity=debug ml-engine jobs \
       submit training $JOB_NAME \
                --package-path=youtube-8m \
                --module-name=youtube-8m.eval \
                --staging-bucket=$BUCKET_NAME \
                --region=us-central1 \
                --config=youtube-8m/cloudml-gpu.yaml \
                -- --eval_data_pattern='gs://youtube8m-ml-us-east1/2/frame/validate/validate*.tfrecord' \
                --model=$MODEL_NAME \
                --train_dir=$TRAIN_DIR/${JOB_TO_EVAL} \
                --run_once=True \
                --frame_features=True \
                --feature_names='rgb,audio' \
                --feature_sizes="1024,128" \
                --batch_size=128 \
                --netvlad_cluster_size=256 \
                --netvlad_hidden_size=1024 \
                --moe_l2=1e-6 \
                --iterations=300 \
                --learning_rate_decay=0.8 \
                --netvlad_relu=False  \
                --gating=True \
                --moe_prob_gating=False \
                --is_training=False \
                --moe_num_mixtures=2

                
