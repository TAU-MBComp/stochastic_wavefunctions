# syntax=docker/dockerfile:1
FROM tensorflow/tensorflow:latest-gpu
RUN pip install sympy
RUN pip install matplotlib
RUN pip install scipy
