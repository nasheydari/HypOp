'''
Test functions for optimization, implemnented based on https://en.wikipedia.org/wiki/Test_functions_for_optimization
'''

"""
This software is Copyright © 2XXX The Regents of the University of California. All Rights Reserved.
Permission to copy, modify, and distribute this software and its documentation for educational, research and
non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the
above copyright notice, this paragraph and the following three paragraphs appear in all copies. Permission
to make commercial use of this software may be obtained by contacting:

Office of Innovation and Commercialization
9500 Gilman Drive, Mail Code 0910
University of California
La Jolla, CA 92093-0910
(858) 534-5815
innovation@ucsd.edu

This software program and documentation are copyrighted by The Regents of the University of California.
The software program and documentation are supplied “as is”, without any accompanying services from
The Regents. The Regents does not warrant that the operation of the program will be uninterrupted or error-
free. The end-user understands that the program was developed for research purposes and is advised not to
rely exclusively on the program for any reason.

IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY FOR
DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION,
EVEN IF THE UNIVERSITY OF CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY OF
SUCH DAMAGE. THE UNIVERSITY OF CALIFORNIA SPECIFICALLY DISCLAIMS ANY
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE
PROVIDED HEREUNDER IS ON AN “AS IS” BASIS, AND THE UNIVERSITY OF CALIFORNIA
HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES,
ENHANCEMENTS, OR MODIFICATIONS.
"""


import numpy as np


def rastrigin(x, A=10):
	x = np.asarray(x)
	f = A*len(x) + np.sum(x**2 - A*np.cos(2*np.pi*x))
	return f

def ackley(x):
	# only 2d
	x = np.asarray(x)
	f = -20*np.exp(-0.2*np.sqrt(0.5*np.sum(x**2))) - np.exp(0.5*np.sum(np.cos(2*np.pi*x))) + np.exp(1.) + 20
	return f

def sphere(x):
	# only 2d
	assert len(x)==2, 'only 2d input vectors accepted for this function'
	x = np.asarray(x)
	f = np.sum(x**2)
	return f

def rosenbrock(x):
	f = 0
	for i in range(len(x)-1):
		f += 100*((x[i+1]-(x[i]**2))**2)+((1-x[i])**2)
	return f

def beale(x):
	# only 2d
	assert len(x)==2, 'only 2d input vectors accepted for this function'
	f = (1.5-x[0]+(x[0]*x[1]))**2 + (2.25-x[0]+(x[0]*(x[1]**2)))**2 + (2.625-x[0]+(x[0]*(x[1]**3)))**2
	return f

def himmelblau(x):
	# only 2d
	assert len(x)==2, 'only 2d input vectors accepted for this function'
	f = (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
	return f
