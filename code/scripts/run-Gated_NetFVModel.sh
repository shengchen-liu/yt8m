# python train.py 
# --train_data_pattern="$path_to_features/*a*??.tfrecord" 
# --model=NetFVModelLF 
# --train_dir=gatednetfvLF-128k-1024-80-0002-300iter-norelu-basic-gatedmoe 
# --frame_features=True 
# --feature_names="rgb,audio" 
# --feature_sizes="1024,128" 
# --batch_size=80 -
# -base_learning_rate=0.0002 --fv_cluster_size=128 --fv_hidden_size=1024 --moe_l2=1e-6 --iterations=300 --learning_rate_decay=0.8 --fv_relu=False --gating=True --moe_prob_gating=True 
# --fv_couple_weights=False --max_step=600000

BUCKET_NAME=gs://${USER}_yt8m_train_bucket
# BUCKET_NAME=gs://${USER}_yt8m_sample_bucket

# (One Time) Create a storage bucket to store training logs and checkpoints.
# gsutil mb -l us-east1 $BUCKET_NAME
# Submit the training job.
JOB_NAME=yt8m_train_$(date +%Y%m%d_%H%M%S)
MODEL_NAME=NetFVModelLF 
TRAIN_DIR=$BUCKET_NAME/all/gatednetfvLF-128k-1024-80-0002-300iter-norelu-basic-gatedmoe

gcloud --verbosity=debug ml-engine jobs submit training $JOB_NAME \
                                                      --package-path=youtube-8m \
                                                      --module-name=youtube-8m.train \
                                                      --staging-bucket=$BUCKET_NAME --region=us-central1 \
                                                      --config=youtube-8m/cloudml-8gpu.yaml \
                                                      -- --train_data_pattern='gs://youtube8m-ml-us-east1/2/frame/train/train*.tfrecord' \
                                                      --frame_features=True \
                                                      --model=$MODEL_NAME \
                                                      --train_dir=$TRAIN_DIR  \
                                                      --feature_names="rgb,audio" \
                                                      --feature_sizes="1024,128" \
                                                      --batch_size=80 \
                                                      --base_learning_rate=0.0002 \
                                                      --learning_rate_decay=0.8 \
                                                      --fv_cluster_size=128 \
                                                      --fv_hidden_size=1024 \
                                                      --moe_l2=1e-6 \
                                                      --iterations=300 \
                                                      --fv_relu=False \
                                                      --gating=True \
                                                      --moe_prob_gating=True \
                                                      --fv_couple_weights=False \
                                                      --num_readers=4 \
                                                      --num_epochs=5 