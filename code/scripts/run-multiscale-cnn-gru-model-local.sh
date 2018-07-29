TRAIN_DIR="yt8m_train_local/all/multiscale_cnn_gru_model_batch128_decay08" 
MODEL_NAME=MultiscaleCnnGruModel
gcloud --verbosity=debug ml-engine local train --package-path=youtube-8m \
                                               --module-name=youtube-8m.train \
                                               -- --train_data_pattern='gs://youtube8m-ml-us-east1/2/frame/train/train*.tfrecord' \
                                               --frame_features=True \
                                               --model=$MODEL_NAME \
                                               --train_dir=$TRAIN_DIR  \
                                               --feature_names="rgb,audio" \
                                               --feature_sizes="1024,128" \
                                               --multiscale_cnn_gru_layers=4 \
                                               --moe_num_mixtures=4 \
                                               --is_training=True \
                                               --batch_size=128 \
                                               --base_learning_rate=0.001 \
                                               --learning_rate_decay=0.8 \
                                               --train_dir=$TRAIN_DIR  \
                                               --num_readers=4 \
                                               --num_epochs=5 \
                                               --start_new_model=True
                                               


