for file in config_autogenerated/*; do
    sh distributed_pipeline.sh "${file}"
done