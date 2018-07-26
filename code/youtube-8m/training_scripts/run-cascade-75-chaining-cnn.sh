CUDA_VISIBLE_DEVICES=0 python train.py \
	--train_dir="../model/distillchain_cnn_dcc" \
	--train_data_pattern="/Youtube-8M/distillation/frame/train/*.tfrecord" \
	--frame_features=True \
	--feature_names="rgb,audio" \
	--feature_sizes="1024,128" \
	--distillation_features=True \
	--distillation_as_input=True \
	--model=DistillchainCnnDeepCombineChainModel \
  --deep_chain_layers=3 \
  --deep_chain_relu_cells=256 \
  --moe_num_mixtures=4 \
  --multitask=True \
  --label_loss=MultiTaskCrossEntropyLoss \
  --support_type="label,label,label" \
  --support_loss_percent=0.05 \
	--num_readers=4 \
	--batch_size=128 \
	--num_epochs=3 \
  --keep_checkpoint_every_n_hours=0.5 \
	--base_learning_rate=0.001

