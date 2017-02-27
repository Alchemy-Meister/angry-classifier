#!/usr/bin/env bash

makeglossaries repressed-anger-detection
biber repressed-anger-detection
xelatex -shell-escape repressed-anger-detection
