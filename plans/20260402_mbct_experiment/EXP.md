| ID | Init                      | MDT | RawBoost | MBCT | LoRA | Purpose                                          |
| -- | ------------------------- | --- | -------- | ---- | ---- | ------------------------------------------------ |
| A  | scratch / baseline recipe | yes | yes      | no   | no   | Reference, no need to run                        |
| B  | same as A                 | yes | yes      | yes  | no   | Pure MBCT effect                                 |
| C  | pretrained A              | yes | yes      | no   | yes  | Pure LoRA effect, no need to run                 |
| D  | pretrained A              | yes | yes      | yes  | yes  | LoRA + MBCT                                      |
| E  | pretrained A              | no  | no       | yes  | yes  | Current weak setting, no need to run             |