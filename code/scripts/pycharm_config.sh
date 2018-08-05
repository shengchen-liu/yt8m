--train_data_pattern=/media/shengchen/Shengchen/yt8m/input/frame/train*.tfrecord
\
--frame_features=True
\
--model=NetVLADModelLF_LSTM
\
--train_dir=/media/shengchen/Shengchen/yt8m/yt8m_train_local/gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe-lstm 
\
--feature_names="rgb,audio"
\
--feature_sizes="1024,128"
\
--batch_size=1024
\
--multiscale_cnn_lstm_layers=4 \
--moe_num_mixtures=4 \
--is_training=True \
--base_learning_rate=0.001 \
--num_readers=4 \
--num_epochs=5 \
--verbosity=debug \
--start_new_model=True \

--train_data_pattern='gs://youtube8m-ml-us-east1/2/frame/train/train*.tfrecord' \
--frame_features=True \
--model=MultiscaleCnnGruModel \
--train_dir=yt8m_train_local/all/multiscale_cnn_gru_model_batch128_decay08  \
--feature_names="rgb,audio" \
--feature_sizes="1024,128" \
-multiscale_cnn_gru_layers=4 \
--moe_num_mixtures=4 \
--is_training=True \
--batch_size=128 \
--base_learning_rate=0.001 \
--learning_rate_decay=0.8 \
--num_readers=4 \
--num_epochs=5 \