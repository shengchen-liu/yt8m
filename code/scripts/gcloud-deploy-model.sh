MODEL="sentiment_model"
MODEL_BINARIES="gs://lawson_movie/sentiment_keras_cloud_ml/run1531246736.11/export/"
VERSION="v4"

gcloud ml-engine versions create $VERSION --model $MODEL --origin $MODEL_BINARIES --runtime-version 1.8