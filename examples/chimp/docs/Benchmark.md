# facial-landmark-benchmark

## Usage

### Run detection

```
python benchmark.py \
    --input-dir ./data/Chimp_2 \
    --output-dir ./data/Chimp_2_ft_mouth800 \
    --model-name ft_mouth800 \
    --model-path ./model/ft_mouth800.ckpt \
    --format png \
    --flag-detect
```

or pass `--flag-save` to save the augmented images.

```
python benchmark.py \
    --input-dir ./data/Chimp_2 \
    --output-dir ./data/Chimp_2_ft_mouth800 \
    --model-name ft_mouth800 \
    --model-path ./model/ft_mouth800.ckpt \
    --format png \
    --flag-save
```

### Run evaluation

```
python evaluate.py \
    --input-dir ./data/Chimp_2_ft_mouth800 \
    --model-name ft_mouth800
```

### Run visualization

```
python visualize.py \
    --input-dir ./data/Chimp_2_ft_mouth800 \
    --model-name ft_mouth800
```
