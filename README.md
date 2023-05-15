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
- [Dataset]()
- [Pre-trained Model]()
- [Fine-tuned Model]()

## Run
If you want to use our model for training, you can directly use the following command:
```shell
bash run_clebpi.sh
```
Note: Before running, you need to config two paths: 1) dataset path; 2) model path.

Here are some useful parameters:
- `do_train`: Whether fine-tune model
- `do_eval`: Whether perform evaluation on validation set during the training
- `do_predict`: Whether performing prediction on the test set when finishing training, if you only want to use our model to make predictions, you can set it to true and set `do_train` and `do_eval` to false.
- `max_seq_length`: The truncation length. If the input length is longer than it, we will keep the first `max_seq_length` tokens in the input length.
