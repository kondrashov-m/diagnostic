#!/bin/bash

OUT_FOLDER_PATH="out_files"
INPUT_FOLDER_PATH="input_files"

mkdir -p "$OUT_FOLDER_PATH"
mkdir -p "$INPUT_FOLDER_PATH"

shopt -s nullglob

for file in "$INPUT_FOLDER_PATH"/*; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        
        filename_no_ext="${filename%.*}"
        extension="${filename##*.}"

        mkdir -p "$OUT_FOLDER_PATH/$filename_no_ext"
        ffmpeg -i "$file" -f segment -segment_time 60 -c copy "$OUT_FOLDER_PATH/$filename_no_ext/output_%03d.$extension"
    fi
done