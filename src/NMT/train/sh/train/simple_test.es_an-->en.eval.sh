python -m NMT.train.train \
    -c "src/configs/test.es_an-->en.yaml" \
    -m EVAL \
    --nmt_corpus child \
    --HPC \
    --model_name little-fancy-hedgehog-of-science > "src/NMT/train/sh/train/simple_test.es_an-->en.eval.out"