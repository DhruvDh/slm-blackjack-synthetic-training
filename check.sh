#!/bin/bash

tail -n 1 train-2048.err && echo -e "\n" && tail -n 1 train-4096.err && echo -e "\n" && tail -n 1 train-8192.err && echo -e "\n"
