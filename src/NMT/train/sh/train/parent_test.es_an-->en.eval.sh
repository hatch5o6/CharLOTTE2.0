python -m NMT.train.train \
    -c "src/configs/test.es_an-->en.yaml" \
    -m EVAL \
    --nmt_corpus parent \
    --HPC \
    --model_name meticulous-tangible-lizard-from-asgard > "src/NMT/train/sh/train/parent_test.es_an-->en.eval.out"