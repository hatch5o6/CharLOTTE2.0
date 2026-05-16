python -m NMT.train.train \
    -c "src/configs/test.es_an-->en.yaml" \
    -m EVAL \
    --nmt_corpus child \
    --fine_tune \
    --HPC > "src/NMT/train/sh/child_test.es_an-->en.eval.out"