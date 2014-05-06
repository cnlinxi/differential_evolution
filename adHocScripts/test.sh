#!/bin/sh
#$ -l mem=32G
#$ -l rmem=32G
#$ -j yes
#$ -l h_rt=48:00:00
abaqus cae noGUI=bloodhoundProblem.py
