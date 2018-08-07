BUCKET_NAME=gs://${USER}_yt8m_train_bucket
TRAIN_DIR=gs://${USER}_yt8m_train_bucket
JOB_TO_INFER=yt8m_train_video_level_MoeModel/epoch8
JOB_NAME=yt8m_inference_$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="gs://${USER}_yt8m_train_bucket/all/model_predictions/${JOB_TO_INFER}"
MODEL_NAME=MoeModel
gcloud --verbosity=debug ml-engine jobs \
       submit training $JOB_NAME \
              --package-path=youtube-8m \
              --module-name=youtube-8m.inference \
              --staging-bucket=$BUCKET_NAME \
              --region=us-central1 \
              --config=youtube-8m/cloudml-gpup100.yaml \
              -- --input_data_pattern='gs://youtube8m-ml/2/video/test/test*.tfrecord' \
              --output_dir=$OUTPUT_DIR\
              --train_dir=$BUCKET_NAME/${JOB_TO_INFER} \
              --model=$MODEL_NAME \
              --frame_features=False \
             --feature_names="mean_rgb,mean_audio" \
             --feature_sizes="1024,128" \
             --batch_size=512 \
             --num_readers=8 \
             --file_size=4096 \
             --output_file=$OUTPUT_DIR/predictions.csv


# for part in ensemble_train ensemble_validate test; do 
#     CUDA_VISIBLE_DEVICES=0 python inference-pre-ensemble.py \
#              --output_dir="/Youtube-8M/model_predictions/${part}/video_moe16_model" \
#         --model_checkpoint_path="../model/video_moe16_model/model.ckpt-19058" \
#              --input_data_pattern="/Youtube-8M/data/video/${part}/*.tfrecord" \
#              --frame_features=False \
#              --feature_names="mean_rgb,mean_audio" \
#              --feature_sizes="1024,128" \
#         --model=MoeModel \
#         --moe_num_mixtures=16 \
#              --batch_size=128 \
#              --file_size=4096
# done


# gcloud --verbosity=debug ml-engine jobs \
#        submit training $JOB_NAME \
#               --package-path=youtube-8m \
#               --module-name=youtube-8m.inference-pre-ensemble \
#               --staging-bucket=$BUCKET_NAME \
#               --region=us-central1 \
#               --config=youtube-8m/cloudml-4gpu.yaml \
#               -- --input_data_pattern='gs://youtube8m-ml/2/frame/test/test*.tfrecord' \
#               --output_dir=$OUTPUT_DIR\
#              --train_dir=$TRAIN_DIR/${JOB_TO_INFER} \
#              --model=$MODEL_NAME \
#                --frame_features=True \
#                --feature_names="rgb,audio" \
#                --feature_sizes="1024,128" \
#              --batch_size=512 \
#              --num_readers=8 \
#              --file_size=4096