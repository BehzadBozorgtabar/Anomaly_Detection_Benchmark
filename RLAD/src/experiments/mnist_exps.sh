#!/usr/bin/env bash


#mnist training exp1 --w_rec 0.9 --w_svdd 0.1

#for i in 0 1 2 3 4 5 6 7 8 9
#do
#    python main.py mnist mnist_LeNet ../log/mnist_test ../data --seed 1 --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 1e-6 --n_jobs_dataloader 12 --normal_class $i --w_rec 0.9 --w_svdd 0.1
#done


#mnist training exp2 --w_rec 0.75 --w_svdd 0.25

#for i in 0 1 2 3 4 5 6 7 8 9
#do
#    python main.py mnist mnist_LeNet ../log/mnist_test ../data --seed 1 --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 1e-6 --n_jobs_dataloader 12 --normal_class $i --w_rec 0.75 --w_svdd 0.25
#done


#mnist training exp3 --w_rec 0.5 --w_svdd 0.5

#for i in 0 1 2 3 4 5 6 7 8 9
#do
#    python main.py mnist mnist_LeNet ../log/mnist_test ../data --seed 1 --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 1e-6 --n_jobs_dataloader 12 --normal_class $i --w_rec 0.5 --w_svdd 0.5
#done


#mnist training exp4 --w_rec 0.25 --w_svdd 0.75

#for i in 0 1 2 3 4 5 6 7 8 9
#do
#    python main.py mnist mnist_LeNet ../log/mnist_test ../data --seed 1 --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 1e-6 --n_jobs_dataloader 12 --normal_class $i --w_rec 0.25 --w_svdd 0.75
#done


#mnist training exp4 --w_rec 0.1 --w_svdd 0.9

for i in 0 #1 2 3 4 5 6 7 8 9
do
    python main.py mnist mnist_LeNet ../log/mnist_test ../data --seed 1 --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 1e-6 --n_jobs_dataloader 12 --normal_class $i --w_rec 5 --w_svdd 500
done


