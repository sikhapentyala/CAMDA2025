#!/usr/bin/env bash

MODE=non-dp
MODE=dp

num_iters=1000
norm=discretize

if [[ ${MODE} = 'non-dp' ]]
then
    # Non-dp
    python main.py \
        -name aml_${num_iters}/test0.2\
        --test_frac 0.2 \
        --preprocess ${norm}\
        --if_filter_x\
        --random_seed 1000\
        --preprocess_arg alpha=0.25\
        -iters ${num_iters}\
        --if_evaluate
fi


if [[ ${MODE} = 'dp' ]]
then
    python main.py \
        -name dp_aml_${num_iters}/test0.2 \
        --test_frac 0.2 \
        --preprocess ${norm} \
        --if_filter_x\
        --enable_privacy\
        --target_epsilon 8\
        --if_evaluate\
        --preprocess_arg alpha=0.25\
        --target_delta 1e-5\
        --random_seed 1000\
        -iters ${num_iters}
fi