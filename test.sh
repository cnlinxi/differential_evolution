#!/bin/sh
#$ -l mem=16G
#$ -l rmem=8G
#$ -j yes
abaqus cae noGUI=bloodhoundProblem.py
