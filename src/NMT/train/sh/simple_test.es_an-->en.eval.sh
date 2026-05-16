python -m NMT.train.train \
    -c "src/configs/test.es_an-->en.yaml" \
    -m EVAL \
    --nmt_corpus child \
    --HPC > "src/NMT/train/sh/simple_test.es_an-->en.eval.out"