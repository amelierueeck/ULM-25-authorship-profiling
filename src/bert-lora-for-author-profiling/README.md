---
library_name: peft
license: apache-2.0
base_model: google-bert/bert-base-uncased
tags:
- base_model:adapter:google-bert/bert-base-uncased
- lora
- transformers
model-index:
- name: bert-lora-for-author-profiling
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# bert-lora-for-author-profiling

This model is a fine-tuned version of [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.7884
- Age Acc: 0.5871
- Gender Acc: 0.7004
- Joint Acc: 0.4172

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 9.7145e-05
- train_batch_size: 16
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 2
- total_train_batch_size: 32
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 2

### Training results

| Training Loss | Epoch  | Step  | Validation Loss | Age Acc | Gender Acc | Joint Acc |
|:-------------:|:------:|:-----:|:---------------:|:-------:|:----------:|:---------:|
| 0.9321        | 0.0515 | 1000  | 0.8749          | 0.5348  | 0.6553     | 0.3517    |
| 0.8691        | 0.1031 | 2000  | 0.8534          | 0.5491  | 0.6666     | 0.3685    |
| 0.8526        | 0.1546 | 3000  | 0.8424          | 0.5539  | 0.6742     | 0.3752    |
| 0.8468        | 0.2062 | 4000  | 0.8336          | 0.5590  | 0.6784     | 0.3833    |
| 0.8459        | 0.2577 | 5000  | 0.8280          | 0.5636  | 0.6821     | 0.3865    |
| 0.837         | 0.3093 | 6000  | 0.8242          | 0.5637  | 0.6849     | 0.3891    |
| 0.8316        | 0.3608 | 7000  | 0.8193          | 0.5680  | 0.6875     | 0.3937    |
| 0.8302        | 0.4124 | 8000  | 0.8177          | 0.5703  | 0.6857     | 0.3952    |
| 0.824         | 0.4639 | 9000  | 0.8147          | 0.5706  | 0.6885     | 0.3969    |
| 0.8214        | 0.5155 | 10000 | 0.8122          | 0.5728  | 0.6897     | 0.3992    |
| 0.8206        | 0.5670 | 11000 | 0.8097          | 0.5735  | 0.6925     | 0.4013    |
| 0.8172        | 0.6185 | 12000 | 0.8073          | 0.5736  | 0.6939     | 0.4021    |
| 0.8152        | 0.6701 | 13000 | 0.8059          | 0.5750  | 0.6941     | 0.4032    |
| 0.8148        | 0.7216 | 14000 | 0.8047          | 0.5769  | 0.6955     | 0.4058    |
| 0.8121        | 0.7732 | 15000 | 0.8022          | 0.5776  | 0.6952     | 0.4057    |
| 0.8139        | 0.8247 | 16000 | 0.8003          | 0.5791  | 0.6967     | 0.4076    |
| 0.8081        | 0.8763 | 17000 | 0.8011          | 0.5788  | 0.6962     | 0.4085    |
| 0.8084        | 0.9278 | 18000 | 0.7993          | 0.5787  | 0.6965     | 0.4084    |
| 0.8014        | 0.9794 | 19000 | 0.7981          | 0.5812  | 0.6961     | 0.4105    |
| 0.8029        | 1.0309 | 20000 | 0.8004          | 0.5809  | 0.6942     | 0.4075    |
| 0.8035        | 1.0824 | 21000 | 0.7972          | 0.5813  | 0.6978     | 0.4112    |
| 0.8001        | 1.1340 | 22000 | 0.7961          | 0.5823  | 0.6982     | 0.4129    |
| 0.8029        | 1.1855 | 23000 | 0.7947          | 0.5817  | 0.6990     | 0.4122    |
| 0.8008        | 1.2371 | 24000 | 0.7924          | 0.5844  | 0.7001     | 0.4142    |
| 0.8046        | 1.2886 | 25000 | 0.7968          | 0.5844  | 0.6944     | 0.4110    |
| 0.7987        | 1.3401 | 26000 | 0.7916          | 0.5842  | 0.7003     | 0.4147    |
| 0.7963        | 1.3917 | 27000 | 0.7922          | 0.5842  | 0.6996     | 0.4148    |
| 0.8038        | 1.4432 | 28000 | 0.7909          | 0.5841  | 0.7017     | 0.4149    |
| 0.7952        | 1.4948 | 29000 | 0.7924          | 0.5847  | 0.6984     | 0.4135    |
| 0.7974        | 1.5463 | 30000 | 0.7913          | 0.5856  | 0.6993     | 0.4155    |
| 0.8019        | 1.5979 | 31000 | 0.7902          | 0.5857  | 0.7000     | 0.4159    |
| 0.8012        | 1.6494 | 32000 | 0.7889          | 0.5861  | 0.7014     | 0.4170    |
| 0.7953        | 1.7010 | 33000 | 0.7895          | 0.5858  | 0.7009     | 0.4165    |
| 0.7996        | 1.7525 | 34000 | 0.7887          | 0.5866  | 0.7011     | 0.4168    |
| 0.7944        | 1.8041 | 35000 | 0.7884          | 0.5868  | 0.7012     | 0.4170    |
| 0.7961        | 1.8556 | 36000 | 0.7883          | 0.5866  | 0.7009     | 0.4173    |
| 0.7955        | 1.9071 | 37000 | 0.7876          | 0.5869  | 0.7011     | 0.4175    |
| 0.7961        | 1.9587 | 38000 | 0.7884          | 0.5871  | 0.7004     | 0.4172    |


### Framework versions

- PEFT 0.17.1
- Transformers 4.55.2
- Pytorch 2.6.0+cu124
- Datasets 3.6.0
- Tokenizers 0.21.4