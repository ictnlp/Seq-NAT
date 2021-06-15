data_dir=data-bin/wmt14_en_de_distill
save_dir=output/pretrain
CUDA_VISIBLE_DEVICES=0 python train.py $data_dir \
    --src-embedding-copy --fp16 --ddp-backend=no_c10d --save-dir $save_dir \
    --task translation_lev \
    --criterion nat_loss \
    --arch nonautoregressive_transformer \
    --noise full_mask \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9,0.98)'  \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --min-lr '1e-09' --warmup-updates 10000 \
    --warmup-init-lr '1e-07' --label-smoothing 0.1 \
    --dropout 0.3 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --pred-length-offset \
    --length-loss-factor 0.1 \
    --apply-bert-init \
    --log-format 'simple' --log-interval 100 \
    --max-tokens 8000 --update-freq 16\
    --save-interval-updates 5000 \
    --max-update 300000 --keep-interval-updates 10 --no-epoch-checkpoints
