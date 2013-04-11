'''
CS181 HW4
Harvard Spring 2013
-------------------
Joshua Lee and Salvatore Rinchiera
'''

import operator
from math import sqrt
from pylab import *
import matplotlib.pyplot as plt
from scipy.stats import norm, histogram
import numpy.random as rand

def plot_histogram(data, title, jpg_name, display=False, bins=50, width=0.5):
  num_bins = bins
  counts, xaxis = np.histogram(data, bins=num_bins)
  
  plt.figure()
  plt.title(title)
  plt.bar(xaxis[:num_bins], counts, width=width)
  savefig(jpg_name)
  if display: plt.show()

def do_plot(display=False): 
  range = np.arange(-10, 10, 0.001)
  f = 0.2 * norm.pdf(range, 1, sqrt(25)) + 0.3 * norm.pdf(range, -2, 1) \
    + 0.5 * norm.pdf(range, 3, 2) 
  plt.title("plot of f(x) (density function)")
  plt.plot(range, f)
  savefig("plot.jpg")
  if display: plt.show()

def do_direct_sampling(display=False):
  f_samples = []
  num_samples = 500
  uniform_samples = rand.uniform(0, 1, num_samples)
  for u in uniform_samples:
    # choose which normal to sample from
    if u < 0.2:
      new_sample = rand.normal(1, sqrt(25), 1)[0]
    elif u > 0.2 and u < 0.5:
      new_sample = rand.normal(-2, 1, 1)[0]
    else:
      new_sample = rand.normal(3, 2, 1)[0]
  
    f_samples.append(new_sample)
  
  plot_histogram(f_samples, "Histogram of 500 direct samples from f", "direct.jpg", display=display)
  
def do_rejection_sampling(display=False):
  range = np.arange(-10, 10, 0.001)
  f = 0.2 * norm.pdf(range, 1, sqrt(25)) + 0.3 * norm.pdf(range, -2, 1) \
    + 0.5 * norm.pdf(range, 3, 2) 
  
  c = 1.7
  q_mew, q_sd = 0, 4
  q = c * norm.pdf(range, q_mew, q_sd)
  
  plt.figure()
  plt.title("Upper Bounding Function (q)")
  f_plot, = plt.plot(range, f, color='b')
  q_plot, = plt.plot(range, q, color='r')
  plt.legend((f_plot, q_plot), ("f(x)", "c * q(x)"), "upper right")
  savefig("upperbound.jpg")
  if display: plt.show()

  # define our PDFs
  q_pdf = norm(q_mew, q_sd).pdf  
  f1_pdf = norm(1, sqrt(25)).pdf
  f2_pdf = norm(-2, 1).pdf
  f3_pdf = norm(3, 2).pdf
  
  def f_pdf(x):
    return 0.2 * f1_pdf(x) + 0.3 * f2_pdf(x) + 0.5 * f3_pdf(x)

  samples = []
  num_samples = 500
  num_rejects = 0
  num_accepts = 0
  while num_accepts < num_samples:
    # draw a proposal
    x = rand.normal(q_mew, q_sd, 1)[0]
    
    # determine if we should accept or reject
    u = rand.uniform(0, c * q_pdf(x), 1)[0]
    if u < f_pdf(x):
      num_accepts += 1
      samples.append(x)
    else:
      num_rejects += 1
  
  print "Rejection Sampling"
  print "\tnum_rejects = {0}".format(num_rejects)
  
  plot_histogram(samples, "Histogram of 500 rejection samples", "rejection.jpg", display=display)
  
def do_mh(display=False):
  f1_pdf = norm(1, sqrt(25)).pdf
  f2_pdf = norm(-2, 1).pdf
  f3_pdf = norm(3, 2).pdf
  
  def f_pdf(x):
    return 0.2 * f1_pdf(x) + 0.3 * f2_pdf(x) + 0.5 * f3_pdf(x)
  
  mew, sigma = 0, 100
  samples = []
  num_accepts = 0
  num_samples = 500
  x = rand.normal(mew, sigma, 1)[0] # initialize with a direct sample
  for i in range(num_samples):
    x_prime = x + rand.normal(mew, sigma, 1)[0]
    
    # p(x' -> x) / p(x -> x') = 1 b/c normal is symmetric
    alpha = (f_pdf(x_prime)/f_pdf(x))
    
    u = rand.uniform(0, 1, 1)[0]
    
    if u < alpha:
      # accept!
      samples.append(x_prime)
      x = x_prime
      num_accepts += 1
    else:
      # reject :(
      samples.append(x)
  
  print "Metropolis Hastings"
  print "\taccept_rate = {0}".format(float(num_accepts)/num_samples)
  
  plot_histogram(samples, "Histogram of 500 Metropolis Hastings Samples with SD={0}".format(sigma), "mh_sd-{0}.jpg".format(sigma), display=display, bins=50, width=0.2)

if __name__ == "__main__":
  do_plot()
  do_direct_sampling()
  do_rejection_sampling()
  do_mh()
