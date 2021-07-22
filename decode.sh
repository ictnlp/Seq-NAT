model=$1
CUDA_VISIBLE_DEVICES=4 python generate.py data-bin/wmt14_en_de_distill \
    --gen-subset test \
    --task translation_lev \
    --iter-decode-max-iter  0  \
    --iter-decode-eos-penalty 0 \
    --path $model \
    --beam 1  \
    --left-pad-source False \
    --batch-size 1 > out
# because fairseq's output is unordered, we need to recover its order
grep ^H out | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > out.de
python dedup.py
sed -r 's/(@@ )|(@@ ?$)//g' out.de.dedup > out.de.dedup.debpe
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < out.de.dedup.debpe > pred.de
perl multi-bleu.perl ref.de < pred.de
