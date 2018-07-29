BUCKET_NAME=gs://${USER}_yt8m_train_bucket

JOB_TO_EVAL=gatedlightvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe
JOB_NAME=yt8m_inference_$(date +%Y%m%d_%H%M%S)
gcloud --verbosity=debug ml-engine jobs \
       submit training $JOB_NAME \
              --package-path=youtube-8m \
              --module-name=youtube-8m.inference \
              --staging-bucket=$BUCKET_NAME \
              --region=us-east1 \
              --config=youtube-8m/cloudml-gpu.yaml \
              -- --input_data_pattern='gs://youtube8m-ml/2/frame/test/test*.tfrecord' \
              --train_dir=$BUCKET_NAME/all/${JOB_TO_EVAL} \
              --num_readers=8 \
              --output_file=$BUCKET_NAME/${JOB_TO_EVAL}/predictions_netvlad.csv

# python inference.py --output_file=test-gatedlightvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe.csv --input_data_pattern="$path_to_features/test*.tfrecord" --model=NetVLADModelLF --train_dir=gatedlightvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe --frame_features=True --feature_names="rgb,audio" --feature_sizes="1024,128" --batch_size=1024 --base_learning_rate=0.0002 --netvlad_cluster_size=256 --netvlad_hidden_size=1024 --moe_l2=1e-6 --iterations=300 --learning_rate_decay=0.8 --netvlad_relu=False --gating=True --moe_prob_gating=True --lightvlad=True --run_once=True  --top_k=50 
