--train_data_pattern=/Users/shengchen/yt8m/v2/frame/validate*.tfrecord
\
--frame_features=True
\
--model=NetVLADModelLF
\
--train_dir=yt8m_train_local/all/gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe
\
--feature_names="rgb,audio"
\
--feature_sizes="1024,128"
\
--batch_size=80
\
--base_learning_rate=0.0002
\
--learning_rate_decay=0.8
\
--netvlad_cluster_size=256
\
--netvlad_hidden_size=1024
\
--moe_l2=1e-6
\
--iterations=300
\
--netvlad_relu=False
\
--gating=True
\
--moe_prob_gating=True
\
--num_readers=4
\
--num_epochs=5 \
--moe_num_mixtures=2


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