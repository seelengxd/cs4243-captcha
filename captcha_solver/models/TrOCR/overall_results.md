# Results

| Model Name                               | Epochs | Batch Size | Learning Rate | Scheduler | Warmup | Accuracy | Complete Match | Extra Notes |
|------------------------------------------|--------|------------|---------------|-----------|--------|----------|----------------|-------------|
| trocr_base_no_train                      | N/A    | N/A        | N/A           | No        | No     | 0.52 (print) / 0.42 (hand) | 330/1967 (print), 219/1967 (hand) | No additional training, base TrOCR with handwritten pretraining |
| trocr_finetune_print_5epoch_5e-5         | 5      | 8          | 5e-5          | No        | No     | 0.71     | 0.3 (590/1967) | Results file available, trained for longer but lower accuracy |
| trocr_finetune_print_2epoch_3e-5         | 2      | 8          | 3e-5          | No        | No     | 0.77     | 775/1967       | Fine-tuned on printed base, 6k train, 2k eval (likewise for all) |
| trocr_finetune_print_2epoch_1e-5_sched   | 2      | 8          | 1e-5          | Yes       | No     | 0.83     | 994/1967       | Includes scheduler, no warmup |
| trocr_finetune_print_3epoch_5e-6_warmup10 | 3      | 8          | 5e-6          | Yes       | 10%    | 0.84     | 990/1967       | 2nd epoch 0.81, forgot to record complete match |
