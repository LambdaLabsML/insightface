# facial-landmark-benchmark

## Usage

### Run detection

```
python benchmark.py \
    --input-dir ./data/Chimp_2 \
    --output-dir ./data/Chimp_2_ft_mouth800 \
    --model-path ./model/ft_mouth800.ckpt \
    --format png \
    --flag-detect
```

Results will be saved to `./data/Chimp_2_ft_mouth800`

or pass `--flag-save` to save the augmented images.

```
python benchmark.py \
    --input-dir ./data/Chimp_2 \
    --output-dir ./data/Chimp_2_ft_mouth800 \
    --model-path ./model/ft_mouth800.ckpt \
    --format png \
    --flag-save
```

Results will be saved to `./data/Chimp_2_ft_mouth800/imgs_aug`

### Run evaluation

```
python evaluate.py \
    --input-dir ./data/Chimp_2_ft_mouth800
```

Results will be saved to `./data/Chimp_2_ft_mouth800/std.pickle`

### Run visualization

```
python visualize.py \
    --input-dir ./data/Chimp_2_ft_mouth800
```

Results will be saved to `./data/Chimp_2_ft_mouth800/imgs_aggr`
