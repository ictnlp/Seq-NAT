modelfile=$1
python scripts/average_checkpoints.py --inputs $modelfile/ \
 --num-update-checkpoints 5 --output $modelfile/average-model.pt
