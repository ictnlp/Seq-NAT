data_dir=data-bin/wmt14_en_de_distill
mkdir output/rl_traverse
cp output/pretrain/checkpoint_last.pt output/rl_traverse/
save_dir=output/rl_traverse
CUDA_VISIBLE_DEVICES=0 python train.py $data_dir \
    --src-embedding-copy --ddp-backend=no_c10d --save-dir $save_dir \
    --task translation_lev \
    --criterion nat_seq_loss \
    --use-rl --rl-type traverse  --reset-optimizer\
    --arch nonautoregressive_transformer \
    --noise full_mask \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.0001 --lr-scheduler inverse_sqrt \
    --min-lr '1e-09' --warmup-updates 500 \
    --warmup-init-lr '1e-07' --label-smoothing 0.1 \
    --dropout 0.1 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --pred-length-offset \
    --length-loss-factor 0.1 \
    --apply-bert-init \
    --log-format 'simple' --log-interval 1 \
    --max-tokens 4000 --update-freq 32\
    --save-interval-updates 200 \
    --max-update 3000 --keep-interval-updates 10 --no-epoch-checkpoints
