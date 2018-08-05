TRAIN_DIR="/media/shengchen/Shengchen/yt8m/yt8m_train_local/parallel-lstm-audio"
# TRAIN_DIR='gs://shengchen_yt8m_sample_bucket_1/yt8m/v2/frame/train*.tfrecord'
MODEL_NAME=LstmModel
gcloud ml-engine local train --package-path=youtube-8m \
                             --module-name=youtube-8m.train \
                             -- --train_data_pattern=TRAIN_DIR \
                             --frame_features=True \
							 --model=$MODEL_NAME \
	                         --feature_names="audio" \
	                         --feature_sizes="128" \
	                         --multiscale_cnn_lstm_layers=4 \
	                         --moe_num_mixtures=4 \
	                         --multitask=True \
	                         --label_loss=MultiTaskCrossEntropyLoss \
	                         --support_loss_percent=1.0 \
	                         --support_type="label,label,label,label" \
	                         --is_training=True \
	                         --batch_size=64 \
	                         --base_learning_rate=0.001 \
	                         --train_dir=$TRAIN_DIR  \
	                         --num_readers=4 \
	                         --num_epochs=5 \
	                         --start_new_model=True \
                             --verbosity=debug


# BUCKET_NAME=gs://${USER}_yt8m_train_bucket
# # BUCKET_NAME=gs://${USER}_yt8m_sample_bucket

# # (One Time) Create a storage bucket to store training logs and checkpoints.
# # gsutil mb -l us-east1 $BUCKET_NAME
# # Submit the training job.
# JOB_NAME=yt8m_train_$(date +%Y%m%d_%H%M%S)
# MODEL_NAME=MultiscaleCnnLstmModel
# TRAIN_DIR=$BUCKET_NAME/all/multiscale_cnn_lstm_model

# gcloud --verbosity=debug ml-engine jobs submit training $JOB_NAME \
#                                                       --package-path=youtube-8m \
#                                                       --module-name=youtube-8m.train \
#                                                       --staging-bucket=$BUCKET_NAME --region=us-central1 \
#                                                       --config=youtube-8m/cloudml-8gpu.yaml \
#                                                       -- --train_data_pattern='gs://youtube8m-ml-us-east1/2/frame/train/train*.tfrecord' \
#                                                       --frame_features=True \
#                                                       --model=$MODEL_NAME \
#                                                       --feature_names="rgb,audio" \
#                                                       --feature_sizes="1024,128" \
#                                                       --multiscale_cnn_lstm_layers=4 \
#                                                       --moe_num_mixtures=4 \
#                                                       --multitask=True \
#                                                       --label_loss=MultiTaskCrossEntropyLoss \
#                                                       --support_loss_percent=1.0 \
#                                                       --support_type="label,label,label,label" \
#                                                       --batch_size=64 \
#                                                       --lstm_cells="1024,128" \
#                                                       --base_learning_rate=0.001 \
#                                                       --train_dir=$TRAIN_DIR  \
#                                                       --num_readers=4 \
#                                                       --num_epoch=5 \
#                                                       --start_new_model=True