#Testing

##Data
Put the dataset in data folder

##Models
Find the models in this [folder](https://drive.google.com/open?id=1bVMpziFjT8rvgBxszZq4Hjo3o4xk_j0I).

1class_model.tar has been trained with one class and default params:

2classes_model.tar has been trained with two classes and default params:

They are the best checkpoint with k=10 as validation has been made with k=10.

##Run
Go into folder src and run:
```
python test.py CXR_author unet ../log/test ../data/author --load_model 'model path' --k (k nearest neighbors to consider) --rep_dim 100
```

