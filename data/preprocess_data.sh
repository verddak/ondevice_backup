SCRIPT="preprocess_data.py"
RAW_DATA_DIR="/home/hai_chanyoung/code/ondevice/ondevice_backup/data/filtered_raw"
OUTPUT_BASE_DIR="/home/hai_chanyoung/code/ondevice/ondevice_backup/data"

# Default: 80/20 split, seed=42
for dataset in "2mm_#1" "5mm_#2"; do
    python $SCRIPT \
        --raw_data_dir "$RAW_DATA_DIR/$dataset" \
        --output_dir "$OUTPUT_BASE_DIR/${dataset}_4class_seed42" \
        --seed 42 \
        --mode split \
        --valid_size 0.2
done