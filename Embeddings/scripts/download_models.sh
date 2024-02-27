#!/bin/bash

cd ..
cd models

# url="https://huggingface.co/lmsys/vicuna-7b-v1.3"
# folder="vicuna-7b-v1.3"
# rm -rf "$folder"
# while true; do
#     git lfs clone $url 
#     if [ $? -eq 0 ]; then
#         echo "$url weight successfully downloaded."
#         break
#     else
#         echo "$url download failed; retrying..."
#     fi
#     rm -rf "$folder"
#     sleep 60s
# done

# url="https://huggingface.co/lmsys/vicuna-13b-v1.3"
# folder="vicuna-13b-v1.3"
# rm -rf "$folder"
# while true; do
#     git lfs clone $url 
#     if [ $? -eq 0 ]; then
#         echo "$url weight successfully downloaded."
#         break
#     else
#         echo "$url download failed; retrying..."
#     fi
#     rm -rf "$folder"
#     sleep 60s
# done

# url="https://huggingface.co/stabilityai/stablelm-tuned-alpha-3b"
# folder="stablelm-tuned-alpha-3b"
# rm -rf "$folder"
# while true; do
#     git lfs clone $url 
#     if [ $? -eq 0 ]; then
#         echo "$url weight successfully downloaded."
#         break
#     else
#         echo "$url download failed; retrying..."
#     fi
#     rm -rf "$folder"
#     sleep 60s
# done

# url="https://huggingface.co/stabilityai/stablelm-tuned-alpha-7b"
# folder="stablelm-tuned-alpha-7b"
# rm -rf "$folder"
# while true; do
#     git lfs clone $url 
#     if [ $? -eq 0 ]; then
#         echo "$url weight successfully downloaded."
#         break
#     else
#         echo "$url download failed; retrying..."
#     fi
#     rm -rf "$folder"
#     sleep 60s
# done

# url="https://huggingface.co/databricks/dolly-v2-3b"
# folder="dolly-v2-3b"
# rm -rf "$folder"
# while true; do
#     git lfs clone $url 
#     if [ $? -eq 0 ]; then
#         echo "$url weight successfully downloaded."
#         break
#     else
#         echo "$url download failed; retrying..."
#     fi
#     rm -rf "$folder"
#     sleep 60s
# done

# url="https://huggingface.co/databricks/dolly-v2-7b"
# folder="dolly-v2-7b"
# rm -rf "$folder"
# while true; do
#     git lfs clone $url 
#     if [ $? -eq 0 ]; then
#         echo "$url weight successfully downloaded."
#         break
#     else
#         echo "$url download failed; retrying..."
#     fi
#     rm -rf "$folder"
#     sleep 60s
# done

# url="https://huggingface.co/databricks/dolly-v2-12b"
# folder="dolly-v2-12b"
# rm -rf "$folder"
# while true; do
#     git lfs clone $url 
#     if [ $? -eq 0 ]; then
#         echo "$url weight successfully downloaded."
#         break
#     else
#         echo "$url download failed; retrying..."
#     fi
#     rm -rf "$folder"
#     sleep 60s
# done

url="https://huggingface.co/openlm-research/open_llama_3b"
folder="open_llama_3b"
rm -rf "$folder"
while true; do
    git lfs clone $url 
    if [ $? -eq 0 ]; then
        echo "$url weight successfully downloaded."
        break
    else
        echo "$url download failed; retrying..."
    fi
    rm -rf "$folder"
    sleep 60s
done

# url="https://huggingface.co/openlm-research/open_llama_7b"
# folder="open_llama_7b"
# rm -rf "$folder"
# while true; do
#     git lfs clone $url 
#     if [ $? -eq 0 ]; then
#         echo "$url weight successfully downloaded."
#         break
#     else
#         echo "$url download failed; retrying..."
#     fi
#     rm -rf "$folder"
#     sleep 10s
# done

# url="https://huggingface.co/THUDM/chatglm-6b"
# folder="chatglm-6b"
# rm -rf "$folder"
# while true; do
#     git lfs clone $url 
#     if [ $? -eq 0 ]; then
#         echo "$url weight successfully downloaded."
#         break
#     else
#         echo "$url download failed; retrying..."
#     fi
#     rm -rf "$folder"
#     sleep 10s
# done
