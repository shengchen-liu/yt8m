TRAIN_DIR="/media/shengchen/Shengchen/yt8m/yt8m_train_local/parallel-lstm-audio"
# TRAIN_DIR='gs://shengchen_yt8m_sample_bucket_1/yt8m/v2/frame/train*.tfrecord'

INPUT="/media/shengchen/Shengchen/yt8m/input/frame/train*.tfrecord"

MODEL_NAME=LstmAudioModel
gcloud ml-engine local train --package-path=youtube-8m \
                             --module-name=youtube-8m.train \
                             -- --train_data_pattern=$INPUT \
                             --frame_features=True \
							 --model=$MODEL_NAME \
	                         --feature_names="video" \
	                         --feature_sizes="1024" \
	                         --multiscale_cnn_lstm_layers=4 \
	                         --moe_num_mixtures=4 \
	                         --is_training=True \
	                         --batch_size=1024 \
	                         --base_learning_rate=0.001 \
	                         --train_dir=$TRAIN_DIR  \
	                         --num_readers=4 \
	                         --num_epochs=5 \
                             --verbosity=debug \
 	                         --start_new_model=True \
