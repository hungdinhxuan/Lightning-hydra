Based on replay_multiview_datamodule, help me implement new strategy replay_balance_class_multiview_datamodule:
- So current replay-based approach each mini-batch ratio from novel set and replay set is taken like 50:50 (new:old)
- For new replay_balance_class: each mini-batch ratio from novel set and replay set is taken like 50:50 (new:old) but make sure that ratio of class is balance
For example: batch size is 8, 50% replay is 4, 50% novel is 4
in 4 novel samples (3 spoof, 1 bonafide) then replay ratio must be same
