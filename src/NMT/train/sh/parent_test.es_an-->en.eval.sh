python -m NMT.train.train \
    -c "src/configs/test.es_an-->en.yaml" \
    -m EVAL \
    --nmt_corpus parent \
    --HPC > "src/NMT/train/sh/parent_test.es_an-->en.eval.out"