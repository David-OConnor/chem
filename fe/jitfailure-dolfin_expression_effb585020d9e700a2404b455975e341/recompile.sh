#!/bin/bash
# Execute this file to recompile locally
/home/david/anaconda3/envs/chem/bin/x86_64-conda_cos6-linux-gnu-c++ -Wall -shared -fPIC -std=c++11 -O3 -fno-math-errno -fno-trapping-math -ffinite-math-only -I/home/david/anaconda3/envs/chem/include -I/home/david/anaconda3/envs/chem/include/eigen3 -I/home/david/anaconda3/envs/chem/.cache/dijitso/include dolfin_expression_effb585020d9e700a2404b455975e341.cpp -L/home/david/anaconda3/envs/chem/lib -L/home/david/anaconda3/envs/chem/home/david/anaconda3/envs/chem/lib -L/home/david/anaconda3/envs/chem/.cache/dijitso/lib -Wl,-rpath,/home/david/anaconda3/envs/chem/.cache/dijitso/lib -lmpi -lmpicxx -lpetsc -lslepc -lz -lhdf5 -lboost_timer -ldolfin -olibdijitso-dolfin_expression_effb585020d9e700a2404b455975e341.so