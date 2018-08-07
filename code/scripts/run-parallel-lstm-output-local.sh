BUCKET_NAME=gs://${USER}_yt8m_train_bucket
MODEL_NAME=LstmParallelFinaloutputModel
TRAIN_DIR=$BUCKET_NAME/all/lstmparallelfinaloutput1024_moe8

gcloud --verbosity=debug ml-engine local train --package-path=youtube-8m \
                                               --module-name=youtube-8m.train \
                                               -- --train_data_pattern='gs://youtube8m-ml-us-east1/2/frame/train/train*.tfrecord' \
                                              --frame_features=True \
                                              --model=$MODEL_NAME \
                                              --feature_names="rgb,audio" \
                                              --feature_sizes="1024,128" \
                                              --moe_num_mixtures=8 \
                                              --batch_size=256 \
                                              --lstm_cells="1024,128" \
                                              --base_learning_rate=0.001 \
                                              --train_dir=$TRAIN_DIR  \
                                              --num_epoch=4 \
                                              --rnn_swap_memory=True \
                                              --is_training=True

                                               

