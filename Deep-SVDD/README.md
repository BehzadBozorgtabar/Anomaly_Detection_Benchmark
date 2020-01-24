#Testing

##Models
Find the models in this [folder](https://drive.google.com/drive/folders/1r3rFdtEtg9GDcBJTfO6t4TWne0tt7w6V?usp=sharing)

CXR_test_40_60 has been trained with:
```
batch_size 32
isize 224
rep_dim 200
w_rec 0.4
w_svdd 0.6
```
CXR_test_50_50 has been trained with:
```
batch_size 32
isize 224
rep_dim 200
w_rec 0.5
w_svdd 0.5
```

CXR_test_60_40 has been trained with:
```
batch_size 32
isize 224
rep_dim 200
w_rec 0.6
w_svdd 0.4
```

##Run
Go into folder src and run:
```
python test.py CXR_author resnet18 ../log/test ../data/author --load_model 'model path' --batch_size 32 --isize 224 --rep_dim 200
```

