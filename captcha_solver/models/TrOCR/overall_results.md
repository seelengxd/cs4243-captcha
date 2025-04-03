# Results

| Model Name                               | Epochs | Batch Size | Learning Rate | Scheduler | Warmup | Accuracy | Complete Match | Extra Notes |
|------------------------------------------|--------|------------|---------------|-----------|--------|----------|----------------|-------------|
| trocr_base_no_train                      | N/A    | N/A        | N/A           | No        | No     | 0.52 (print) / 0.42 (hand) | 330/1967, 0.18 (print), 219/1967, 0.11 (hand) | No additional training, base TrOCR with handwritten/printed pretraining |
| trocr_finetune_print_5epoch_5e-5         | 5      | 8          | 5e-5          | No        | No     | 0.77     | 503/1967, 0.26 | Might be overfitting a lot |
| trocr_finetune_print_2epoch_3e-5         | 2      | 8          | 3e-5          | No        | No     | 0.83     | 808/1967, 0.41      | Fine-tuned on printed base, 6k train, 2k eval (likewise for all) |
| trocr_finetune_print_2epoch_1e-5_sched   | 2      | 8          | 1e-5          | Yes       | No     | 0.88     | 1102/1967, 0.56       | Includes scheduler, no warmup |
| trocr_finetune_print_3epoch_5e-6_warmup10_beam | 3      | 8          | 1e-5         | Yes       | 10%    | 0.89     | 1143/1967, 0.58      | tested beam 1 to 5, slight increase of ~.001 per increase in beam, but all ~0.89, complete match was best at 2 beams with 1143 |
