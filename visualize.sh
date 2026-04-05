# !/bin/bash

# panel 4: done unixcoder-base + use_agnostic_features=true
python visualize.py \
    --config_dir config/config_hybrid.yaml \
    --checkpoint_dir results/unixcoder-base/0/best_model/ \
    --split val \
    --batch_size 32 \
    --perplexity 30 \
    --n_iter 1000 \
    --device_id 0 \
    --output results/unixcoder-base/0/best_model/tsne_val.png

# panel 4: done unixcoder-base + use_agnostic_features=false
python visualize.py \
    --config_dir config/config_hybrid.yaml \
    --checkpoint_dir results/unixcoder-base/1/best_model/ \
    --split val \
    --batch_size 32 \
    --perplexity 30 \
    --n_iter 1000 \
    --device_id 0 \
    --output results/unixcoder-base/1/best_model/tsne_val.png

# panel 2: use_agnostic_features=false + base_model: deberta-v3-base
python visualize.py \
    --config_dir config/config_droiddetect.yaml \
    --checkpoint_dir results/droiddetect-base/0/best_model/ \
    --split val \
    --batch_size 32 \
    --perplexity 30 \
    --n_iter 1000 \
    --device_id 0 \
    --output results/droiddetect-base/0/best_model/tsne_val.png

# train lại ở tmux panel 3: use_agnostic_features=true + base_model: droiddetect-bas <--------------------------------
python visualize.py \
    --config_dir config/config_droiddetect.yaml \
    --checkpoint_dir results/droiddetect-base/1/best_model/ \
    --split val \
    --batch_size 32 \
    --perplexity 30 \
    --n_iter 1000 \
    --device_id 0 \
    --output results/droiddetect-base/1/best_model/tsne_val.png

# panel 5: use_agnostic_features: true + base_model: deberta-v3-base 
python visualize.py \
    --config_dir config/config_hybrid.yaml \
    --checkpoint_dir results/deberta-v3-base/0/best_model/ \
    --split val \
    --batch_size 32 \
    --perplexity 30 \
    --n_iter 1000 \
    --device_id 0 \
    --output results/deberta-v3-base/0/best_model/tsne_val.png

# panel 6: done use_agnostic_features: false + base_model: deberta-v3-base 
python visualize.py \
    --config_dir config/config_hybrid.yaml \
    --checkpoint_dir results/deberta-v3-base/1/best_model/ \
    --split val \
    --batch_size 32 \
    --perplexity 30 \
    --n_iter 1000 \
    --device_id 0 \
    --output results/deberta-v3-base/1/best_model/tsne_val.png