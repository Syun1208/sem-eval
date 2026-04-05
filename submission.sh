# use_agnostic_features: false + base_model: deberta-v3-base: 0.44582
python inference.py --config config/config_hybrid.yaml \
    --test_file ./data/Task_A/test.parquet \
    --checkpoint_dir results/deberta-v3-base/1/best_model/ \
    --batch_size 128 \
    --gpu_id 1,3,5,7 \
    --binary

# use_agnostic_features: true + base_model: deberta-v3-base: 0.28879
python inference.py --config config/config_hybrid.yaml \
    --test_file ./data/Task_A/test.parquet \
    --checkpoint_dir results/deberta-v3-base/0/best_model/ \
    --batch_size 128 \
    --gpu_id 1,3,5,7 \
    --binary

# use_agnostic_features=false + base_model: droiddetect-base: 0.59249
python inference.py --config config/config_droiddetect.yaml \
    --test_file ./data/Task_A/test.parquet \
    --checkpoint_dir results/droiddetect-base/0/best_model/ \
    --batch_size 128 \
    --gpu_id 1,3,5,7 \
    --binary

# use_agnostic_features=true + base_model: droiddetect-base: 
python inference.py --config config/config_droiddetect.yaml \
    --test_file ./data/Task_A/test.parquet \
    --checkpoint_dir results/droiddetect-base/1/best_model/ \
    --batch_size 1024 \
    --gpu_id 1,3,5,7 \
    --binary

# unixcoder-base + use_agnostic_features=false: 0.27839
python inference.py --config config/config_hybrid.yaml \
    --test_file ./data/Task_A/test.parquet \
    --checkpoint_dir results/unixcoder-base/1/best_model/ \
    --batch_size 1024 \
    --gpu_id 1,3,5,7 \
    --binary

# unixcoder-base + use_agnostic_features=true: 0.2214
python inference.py --config config/config_hybrid.yaml \
    --test_file ./data/Task_A/test.parquet \
    --checkpoint_dir results/unixcoder-base/0/best_model/ \
    --batch_size 128 \
    --gpu_id 1,3,5,7 \
    --binary