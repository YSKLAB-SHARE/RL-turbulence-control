# Reinforcement-learning-for-turbulence-control



Calculations are done by importing a Fortran based package compiled by using f2py in a Python scripts.
Fluid computation is done in Fortran and reinforcement learning is done in Python.

Steps to run the code:

1. prepare a flow solver.
The inputs of the solver should include the time steps, initial conditions, and the control input,
and the outputs should contain the velocity fields, skin friction coefficients, the square of the control input,
or other variables for monitoring the calculation, e.g. the bulk velocity etc.

Other inputs/outputs may also necessary for the numerical schemes.

2. compile the solver using f2py.
e.g. f2py -c *.f -m XXX --f77flags='-fopenmp' -lgomp
where XXX is the desired name of the package.
Then, file “XXX.cpython-37m-darwin.so” will be produced.

3. run the python file, call.py.
