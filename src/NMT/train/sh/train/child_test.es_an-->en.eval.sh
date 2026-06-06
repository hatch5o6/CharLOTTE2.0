python -m NMT.train.train \
    -c "src/configs/test.es_an-->en.yaml" \
    -m EVAL \
    --nmt_corpus child \
    --fine_tune \
    --HPC \
    --model_name casual-international-phoenix-of-acceptance > "src/NMT/train/sh/train/child_test.es_an-->en.eval.out"