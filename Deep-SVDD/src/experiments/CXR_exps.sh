#!/usr/bin/env bash

"""
CXR training exp1 --w_rec 0.1 --w_svdd 0.9
"""
for i in 200 500
do
    python main.py CXR_author CXR_resnet18 ../log/CXR_test ../data/author --lr 0.0001 --n_epochs 30 --lr_milestone 100 --batch_size 32 --weight_decay 1e-6 --n_jobs_dataloader 12 --isize 224 --rep_dim $i --w_rec 0.1 --w_svdd 0.9
done

"""
CXR training exp2 --w_rec 0.5 --w_svdd 0.5
"""
for i in 200 500
do
    python main.py CXR_author CXR_resnet18 ../log/CXR_test ../data/author --lr 0.0001 --n_epochs 30 --lr_milestone 100 --batch_size 32 --weight_decay 1e-6 --n_jobs_dataloader 12 --isize 224 --rep_dim $i --w_rec 0.5 --w_svdd 0.5
done

"""
CXR training exp3 --w_rec 0.9 --w_svdd 0.1
"""
for i in 200 500
do
    python main.py CXR_author CXR_resnet18 ../log/CXR_test ../data/author --lr 0.0001 --n_epochs 30 --lr_milestone 100 --batch_size 32 --weight_decay 1e-6 --n_jobs_dataloader 12 --isize 224 --rep_dim $i --w_rec 0.9 --w_svdd 0.1
done

"""
CXR training exp4 --w_rec 0.75 --w_svdd 0.25
"""
for i in 200 500
do
    python main.py CXR_author CXR_resnet18 ../log/CXR_test ../data/author --lr 0.0001 --n_epochs 30 --lr_milestone 100 --batch_size 32 --weight_decay 1e-6 --n_jobs_dataloader 12 --isize 224 --rep_dim $i --w_rec 0.75 --w_svdd 0.25
done

"""
CXR training exp5 --w_rec 0.25 --w_svdd 0.75
"""
for i in 200 500
do
    python main.py CXR_author CXR_resnet18 ../log/CXR_test ../data/author --lr 0.0001 --n_epochs 30 --lr_milestone 100 --batch_size 32 --weight_decay 1e-6 --n_jobs_dataloader 12 --isize 224 --rep_dim $i --w_rec 0.25 --w_svdd 0.75
done
