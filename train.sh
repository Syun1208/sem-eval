# panel 5: done use_agnostic_features: false + base_model: deberta-v3-base 
    python train.py --config config/config_hybrid.yaml \
        --result-dir results/deberta-v3-base/1/ \
        --gpu_id 1,3,5,7

# panel 6: done use_agnostic_features: true + base_model: deberta-v3-base 
python train.py --config config/config_hybrid.yaml \
    --result-dir results/deberta-v3-base/0/ \
    --gpu_id 1,3,5,7

# panel 2: done use_agnostic_features=false + base_model: droiddetect-base
python train.py --config config/config_droiddetect.yaml \
    --result-dir results/droiddetect-base/0/ \
    --gpu_id 1,3,5,7

# panel 3: done use_agnostic_features=true + base_model: droiddetect-base
python train.py --config config/config_droiddetect.yaml \
    --result-dir results/droiddetect-base/1/ \
    --gpu_id 1,3,5,7

# panel 4: done unixcoder-base + use_agnostic_features=false
python train.py --config config/config_hybrid.yaml \
    --result-dir results/unixcoder-base/1/ \
    --gpu_id 1,3,5,7

# panel 1: done unixcoder-base + use_agnostic_features=true
python train.py --config config/config_hybrid.yaml \
    --result-dir results/unixcoder-base/0/ \
    --gpu_id 1,3,5,7