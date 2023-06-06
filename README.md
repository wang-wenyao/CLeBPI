# CLeBPI
This is code implementation of CLeBPI. We test our code on Ubuntu 18.04.

## Dependency
- pytorch == 1.6.0
- transformers == 4.17.0
- datasets == 2.11.0
- tokenizers == 0.13.3
- scikit-learn == 1.2.2
- tqdm

## Download
You need to download dataset, pre-trained model, and fine-tuned model by the following three links:
- [Pt Dataset](https://drive.google.com/file/d/1fq60h8vpcNBwAfxZO5BL4rHe0XpyN277/view?usp=sharing)
- [Ft Dataset](https://drive.google.com/file/d/132PZg5Z0s6tlDz8clPgR-eUXSKPEwj3i/view?usp=sharing)
- [Pre-trained Model](https://drive.google.com/file/d/1bdREEAJCnuE-x7ThB1poE4wnIZpf4GkD/view?usp=sharing)
- [Fine-tuned Model](https://drive.google.com/file/d/1QyP2wzPqMghlQHC5tdoHBf7OC85hwAA1/view?usp=share_link)

## Pre-training
pt folder contains code used for pre-training, including MLM and contrastive learning pre-training. Before runing, you need to config dataset path and model path.

You first to run MLM by the following command:
```shell
bash pt/run_pt.sh
```

When finishing MLM, you can use the following command to perform contrastive learning pre-training:
```shell
bash pt/cl/run.sh
```

## Fine-tuning
If you just want to use our model for fine-tuning, you can directly use the following command:
```shell
bash run_clebpi.sh
```
Note: Before running, you need to config two paths: 1) dataset path; 2) model path. Besides, you also need to download the pre-trained model shared by the above link.

Here are some useful parameters:
- `do_train`: Whether fine-tune model
- `do_eval`: Whether perform evaluation on validation set during the training
- `do_predict`: Whether performing prediction on the test set when finishing training, if you only want to use our model to make predictions, you can set it to true and set `do_train` and `do_eval` to false.
- `max_seq_length`: The truncation length. If the input length is longer than it, we will keep the first `max_seq_length` tokens in the input length.
