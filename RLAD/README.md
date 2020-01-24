#Testing

##Data
Put the dataset in data folder

##Models
The models have been trained in a semi supervised way  
Find the models in this [folder](https://drive.google.com/open?id=1Q1kXtS1u0OmtK-e6fNG0Djtft1zdz-oP).

Autoencoder+neighborhood_clustering+instance_loss has been trained with:
```
batch_size 32 
isize 256
rep_dim 200
lr 0.0001
w_rec 1
w_contrast 0.25
ans_select_rate 0.1 (Run for 1/0.1 = 10 rounds + round 0 for pretraining)
n_epochs 50 (50 epochs per round)
ans_size 10
k 100
```

neighborhood_clustering+instance_loss has been trained with:
```
batch_size 32 
isize 256
rep_dim 200
lr 0.0001
w_rec 0
w_contrast 0.25
ans_select_rate 0.1 (Run for 1/0.1 = 10 rounds + round 0 for pretraining)
n_epochs 50 (50 epochs per round)
ans_size 10
k 100
```

For both, you will find the checkpoints model after each round. 
You will also find the best checkpoint (model.tar) with k=100 as validation has been made with k=100.

##Run
Go into folder src and run:
```
python test.py CXR_author unet ../log/test ../data/author --load_model 'model path' --k (k nearest neighbors to consider)
```

Best score with Autoencoder+neighborhood_clustering+instance_loss at round 10 with k=175

