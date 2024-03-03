
"main.py" is the training file.

"matrixddot_lut.pyx" is a Cython-based app-LUT function, and it needs to be compiled using the command "python setup_dotlut.py build_ext --inplace".

"app_cuda_quant" is a powerful app-LUT calculation tool based on CUDA, and it needs to be compiled using the command "python setup_cuda_app.py install".