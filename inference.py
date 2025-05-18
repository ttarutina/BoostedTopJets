#
# Imports and setup
#
import sys
import matplotlib.pyplot as plt
import numpy as np
import stan
import pandas as pd
from scipy import stats
from scipy.special import softmax
import nest_asyncio
nest_asyncio.apply()
#
np.random.seed(100)
########################################################################
#
# Import data
#
########################################################################
data = np.array(pd.read_csv('data.dat',delimiter=' '))
#
def bin_assigner(bins,val):
#
# This function simply outputs the bin assignment of a given value val for a given choice of binning bins
#
    number_of_bins = len(bins)
    bin_number=0
    if val >= bins[-1]:
        return "Too high"
    if val < bins[0]:
        return "Too low"
    else:
        for bin_number_aux in range(number_of_bins-1):
            if bins[bin_number_aux]<= val < bins[bin_number_aux+1]:
                bin_number = bin_number_aux
        return bin_number
#
#
bins_ncluster = 5
ncluster_bins = np.arange(2.0-0.5,2.0+bins_ncluster+0.5,1.0)
#
bins_mjj = 16
mjj_min = 150
mjj_max = mjj_min+5*(bins_mjj-1)
mjj_bins = np.linspace(mjj_min,mjj_max,bins_mjj)
mjj_vals = np.array([ mjj_min+(_+0.5)*(mjj_max-mjj_min)/(bins_mjj-1) for _ in range(bins_mjj-1)])
#
indexes_ncluster = [a and b for a,b in zip(data[:,0]>=2,data[:,0]<2+bins_ncluster)]
indexes_mjj = [a and b for a,b in zip(data[:,1]>=mjj_min,data[:,1]<mjj_max)]
indexes_full = [a and b for a,b in zip(indexes_ncluster,indexes_mjj)]
data = data[indexes_full]
#
# setting the class fractions of the dataset
#
true_class_fractions = np.array([0.7,0.3])
#
# number of events for the matrix
#
N=70000
#
ncluster = np.zeros(N)
mjj = np.zeros(N)
true_labels = np.zeros(N)
NAB_QCD = np.zeros((bins_mjj-1,bins_ncluster))
NAB_top = np.zeros((bins_mjj-1,bins_ncluster))
#
Nbkg = 0
Nsig = 0
bkg_indexes = np.where(data[:,2]==0)[0]
sig_indexes = np.where(data[:,2]==1)[0]
#
for n in range(N):
    z = np.argmax(stats.multinomial(n=1,p=true_class_fractions).rvs(size=1),axis=1)
    true_labels[n] = z
    if z == 0:
        try: ind_2 = bin_assigner(ncluster_bins,data[int(bkg_indexes[Nbkg]),0])+1
        except: print(bin_assigner(ncluster_bins,data[int(bkg_indexes[Nbkg]),0]))
        try: ind_1 = bin_assigner(mjj_bins,data[int(bkg_indexes[Nbkg]),1])+1
        except: print("Bkg mjj",bin_assigner(mjj_bins,data[int(bkg_indexes[Nbkg]),1]))
        mjj[n]=ind_1
        ncluster[n]=ind_2
        NAB_QCD[ind_1-1,ind_2-1]+=1
        Nbkg+=1
    elif z == 1:
        try: ind_2 = bin_assigner(ncluster_bins,data[int(sig_indexes[Nsig]),0])+1
        except: print(bin_assigner(ncluster_bins,data[int(sig_indexes[Nsig]),0]))
        try: ind_1 = bin_assigner(mjj_bins,data[int(sig_indexes[Nsig]),1])+1
        except: print("Sig mjj",bin_assigner(mjj_bins,data[int(sig_indexes[Nsig]),1]))
        mjj[n]=ind_1
        ncluster[n]=ind_2        
        NAB_top[ind_1-1,ind_2-1]+=1
        Nsig+=1
#
# marginal distributions
#
marginal_QCD_ncluster = np.array([np.sum(ncluster[true_labels==0]==_+1) for _ in range(bins_ncluster)])/np.sum(true_labels==0)
marginal_top_ncluster = np.array([np.sum(ncluster[true_labels==1]==_+1) for _ in range(bins_ncluster)])/np.sum(true_labels==1)
marginal_QCD_mjj = np.array([np.sum(mjj[true_labels==0]==_+1) for _ in range(bins_mjj-1)])/np.sum(true_labels==0)
marginal_top_mjj = np.array([np.sum(mjj[true_labels==1]==_+1) for _ in range(bins_mjj-1)])/np.sum(true_labels==1)
#
########################################################################
#
# EM
#
########################################################################
def E_step(NAB_meas,pa_old,pb_old,Cabcd_old,N1=bins_mjj-1,N2=bins_ncluster):
    gammaabcd=np.zeros((N1,N2,N1,N2))
    for n1 in range(N1):
        for n2 in range(N2):
            for n1_int in range(N1):
                for n2_int in range(N2):
                    gammaabcd[n1,n2,n1_int,n2_int]=Cabcd_old[n1,n2,n1_int,n2_int]*pa_old[n1_int]*pb_old[n2_int]
            gammaabcd[n1,n2,:,:]*=1/(np.sum(gammaabcd[n1,n2,:,:]))
    return gammaabcd
#
#
def M_step(NAB_meas,gammaabcd_old,N1=bins_mjj-1,N2=bins_ncluster):
    N=np.sum(NAB_meas)
    pa_new=np.zeros(N1)
    pb_new=np.zeros(N2)
    Cabcd_new=np.zeros((N1,N2,N1,N2))
    for n1 in range(N1):
        for n2 in range(N2):
            for n2_int in range(N2):
                pa_new+=(NAB_meas[n1,n2]/N)*gammaabcd_old[n1,n2,:,n2_int]

    for n1 in range(N1):
        for n2 in range(N2):
            for n1_int in range(N1):
                pb_new+=(NAB_meas[n1,n2]/N)*gammaabcd_old[n1,n2,n1_int,:]

    for n1_int in range(N1):
        for n2_int in range(N2):
            for n1 in range(N1):
                for n2 in range(N2):
                    Cabcd_new[n1,n2,n1_int,n2_int]=NAB_meas[n1,n2]*gammaabcd_old[n1,n2,n1_int,n2_int]
            Cabcd_new[:,:,n1_int,n2_int]*=1/(np.sum(Cabcd_new[:,:,n1_int,n2_int]))
            
    return pa_new,pb_new,Cabcd_new
#
#
def log_likelihood(NAB_meas,pa_old,pb_old,Cabcd_old,N1=bins_mjj-1,N2=bins_ncluster):
    N=np.sum(NAB_meas)
    log_likelihood_est=0
    for n1 in range(N1):
        for n2 in range(N2):
            internal_likelihood = 0
            for n1_int in range(N1):
                for n2_int in range(N2):
                    internal_likelihood+=Cabcd_old[n1,n2,n1_int,n2_int]*pa_old[n1_int]*pb_old[n2_int]
            #print(n1,n2,internal_likelihood)
            log_likelihood_est+=NAB_meas[n1,n2]*np.log(internal_likelihood)
    return log_likelihood_est
########################################################################
#
# Learning
#
########################################################################
pa_seed = np.sum(NAB_top,1)/np.sum(NAB_top)
pb_seed = np.sum(NAB_top,0)/np.sum(NAB_top)
Cabcd_seed = np.zeros((bins_mjj-1,bins_ncluster,bins_mjj-1,bins_ncluster))
Cabcd_diag = np.zeros((bins_mjj-1,bins_ncluster,bins_mjj-1,bins_ncluster))

for n1 in range(bins_mjj-1):
    for n2 in range(bins_ncluster):
        Cabcd_seed[:,:,n1,n2]=0.1*stats.dirichlet(alpha=np.ones((bins_mjj-1)*bins_ncluster)).rvs()[0].reshape((bins_mjj-1,bins_ncluster))
        Cabcd_seed[n1,n2,n1,n2]+=0.9*1
        Cabcd_diag[n1,n2,n1,n2]=1
#
#
epochs = 100
pa_run_old,pb_run_old,Cabcd_run_old=pa_seed,pb_seed,Cabcd_seed
log_loss = log_likelihood(NAB_top,pa_run_old,pb_run_old,Cabcd_run_old)
for nepoch in range(epochs):
    gammaabcd_new = E_step(NAB_top,pa_run_old,pb_run_old,Cabcd_run_old)
    
    pa_new,pb_new,Cabcd_new = M_step(NAB_top,gammaabcd_new)
    log_loss_new = log_likelihood(NAB_top,pa_new,pb_new,Cabcd_new)
    
    ### add likelihood criteria
    if log_loss_new < log_loss:
        break
    pa_run_old,pb_run_old,Cabcd_run_old = pa_new,pb_new,Cabcd_new
    log_loss = log_loss_new
#
# true distributions and the matrix for tops
#
true_top_ncluster = pb_run_old
true_top_mjj = pa_run_old
Cabcd_top = Cabcd_run_old
#
pa_seed = np.sum(NAB_QCD,1)/np.sum(NAB_QCD)
pb_seed = np.sum(NAB_QCD,0)/np.sum(NAB_QCD)
Cabcd_seed = np.zeros((bins_mjj-1,bins_ncluster,bins_mjj-1,bins_ncluster))
Cabcd_diag = np.zeros((bins_mjj-1,bins_ncluster,bins_mjj-1,bins_ncluster))
#
for n1 in range(bins_mjj-1):
    for n2 in range(bins_ncluster):
        Cabcd_seed[:,:,n1,n2]=0.1*stats.dirichlet(alpha=np.ones((bins_mjj-1)*bins_ncluster)).rvs()[0].reshape((bins_mjj-1,bins_ncluster))
        Cabcd_seed[n1,n2,n1,n2]+=0.9*1
        Cabcd_diag[n1,n2,n1,n2]=1
#
#
epochs = 100
pa_run_old,pb_run_old,Cabcd_run_old=pa_seed,pb_seed,Cabcd_seed
log_loss = log_likelihood(NAB_QCD,pa_run_old,pb_run_old,Cabcd_run_old)
for nepoch in range(epochs):
    gammaabcd_new = E_step(NAB_QCD,pa_run_old,pb_run_old,Cabcd_run_old)
    
    pa_new,pb_new,Cabcd_new = M_step(NAB_QCD,gammaabcd_new)
    log_loss_new = log_likelihood(NAB_QCD,pa_new,pb_new,Cabcd_new)
    
    ### add likelihood criteria
    if log_loss_new < log_loss:
        break
    pa_run_old,pb_run_old,Cabcd_run_old = pa_new,pb_new,Cabcd_new
    log_loss = log_loss_new
#
# true distributions and the matrix for QCD
#
true_QCD_ncluster = pb_run_old
true_QCD_mjj = pa_run_old
Cabcd_QCD = Cabcd_run_old
#
########################################################################
#
# bayesian inference dataset
#
########################################################################
N1 = 100000
ncluster1 = np.zeros(N1)
mjj1 = np.zeros(N1)
true_labels1 = np.zeros(N1)
#
Nbkg1 = Nbkg
Nsig1 = Nsig
#
NAB_QCD_kl = np.zeros((bins_mjj-1,bins_ncluster)) # for KL divergence calc q QCD
NAB_top_kl = np.zeros((bins_mjj-1,bins_ncluster))
#
for n in range(N1):
    z = np.argmax(stats.multinomial(n=1,p=true_class_fractions).rvs(size=1),axis=1)
    true_labels1[n] = z
    if z == 0:
        try: ind_2 = bin_assigner(ncluster_bins,data[int(bkg_indexes[Nbkg1]),0])+1
        except: print(bin_assigner(ncluster_bins,data[int(bkg_indexes[Nbkg1]),0]))
        try: ind_1 = bin_assigner(mjj_bins,data[int(bkg_indexes[Nbkg1]),1])+1
        except: print("Bkg mjj",bin_assigner(mjj_bins,data[int(bkg_indexes[Nbkg1]),1]))
        ncluster1[n] = ind_2
        mjj1[n] = ind_1
        NAB_QCD_kl[ind_1-1,ind_2-1]+=1
        Nbkg1+=1
    elif z == 1:
        try: ind_2 = bin_assigner(ncluster_bins,data[int(sig_indexes[Nsig1]),0])+1
        except: print(bin_assigner(ncluster_bins,data[int(sig_indexes[Nsig1]),0]))
        try: ind_1 = bin_assigner(mjj_bins,data[int(sig_indexes[Nsig1]),1])+1
        except: print("Sig mjj",bin_assigner(mjj_bins,data[int(sig_indexes[Nsig1]),1]))
        ncluster1[n] = ind_2
        mjj1[n] = ind_1
        NAB_top_kl[ind_1-1,ind_2-1]+=1
        Nsig1+=1
#
NAB_QCD_kl = NAB_QCD_kl/(Nbkg1 - Nbkg) # we need it for KL
NAB_top_kl = NAB_top_kl/(Nsig1 - Nsig)
NAB_kl = (NAB_QCD_kl + NAB_top_kl)/(Nbkg1 + Nsig1 - Nbkg - Nsig)
#
# marginal distributions for inference dataset
#
marginal_QCD_ncluster1 = np.array([np.sum(ncluster1[true_labels1==0]==_+1) for _ in range(bins_ncluster)])/np.sum(true_labels1==0)
marginal_top_ncluster1 = np.array([np.sum(ncluster1[true_labels1==1]==_+1) for _ in range(bins_ncluster)])/np.sum(true_labels1==1)
marginal_QCD_mjj1 = np.array([np.sum(mjj1[true_labels1==0]==_+1) for _ in range(bins_mjj-1)])/np.sum(true_labels1==0)
marginal_top_mjj1 = np.array([np.sum(mjj1[true_labels1==1]==_+1) for _ in range(bins_mjj-1)])/np.sum(true_labels1==1)
########################################################################
#
# Stan Model
#
########################################################################
#
# setting the priors (prior vi)
#
########################################################################
Sigma = 1400
myprior_top_mjj = []
mydiff = np.zeros(10)
for index in [3,4,5,6]:
    mydiff[index] = marginal_top_mjj[index]-0.115
mydiff1 = (np.sum(mydiff))/11.
for index in range(15):
    if index in [3,4,5,6]:
        myprior_top_mjj.append(marginal_top_mjj[index]-mydiff[index])
    else:
        myprior_top_mjj.append(marginal_top_mjj[index]+mydiff1)
myprior_top_mjj=np.array(myprior_top_mjj)
#
myprior_QCD_mjj = []
for index in range(15):
    myz = np.argmax(stats.multinomial(n=1,p=[0.5,0.5]).rvs(size=1),axis=1)
    print("myz = ",myz)
    if myz == 1:
       myprior_QCD_mjj.append(0.1-index*0.0033)
    else:
       myprior_QCD_mjj.append(0.1-index*0.0033)
myprior_QCD_mjj = myprior_QCD_mjj/np.sum(myprior_QCD_mjj)
#
eta_ncluster_QCD = Sigma*marginal_QCD_ncluster
eta_ncluster_top = Sigma*marginal_top_ncluster
eta_mjj_QCD = Sigma*myprior_QCD_mjj
eta_mjj_top = Sigma*myprior_top_mjj
eta_class_fractions = np.array([1,1])
#
nprior_samples = 100
prior_samples = [[] for _ in range(nprior_samples)]
for nprior_sample in range(nprior_samples):
    prior_samples[nprior_sample].append(stats.dirichlet(alpha=eta_ncluster_QCD).rvs()[0])
    prior_samples[nprior_sample].append(stats.dirichlet(alpha=eta_ncluster_top).rvs()[0])
    prior_samples[nprior_sample].append(stats.dirichlet(alpha=eta_mjj_QCD).rvs()[0])
    prior_samples[nprior_sample].append(stats.dirichlet(alpha=eta_mjj_top).rvs()[0])
    prior_samples[nprior_sample].append(stats.dirichlet(alpha=eta_class_fractions).rvs()[0])
#
# STAN model code
#
model_conditional_independence = """
data {
  int<lower=1> dncluster;  // bins for ncluster
  int<lower=1> dmjj;  // bins for mjj
  int<lower=1> N1;  // data points
  int<lower=1, upper=dncluster> ncluster1[N1];  // ncluster measurements
  int<lower=1, upper=dmjj> mjj1[N1];  // mjj measurements
  vector[2] eta_class_fractions; // prior for class fractions
  vector[dncluster-1] eta_ncluster_QCD; // prior parameters for QCD ncluster distr.
  vector[dncluster-1] eta_ncluster_top; // prior parameters for top ncluster distr.
  vector[dmjj-1] eta_mjj_QCD; // prior parameters for QCD mjj distr.
  vector[dmjj-1] eta_mjj_top; // prior parameters for top mjj distr.
  array[dmjj-1, dncluster-1,dmjj-1, dncluster-1] real<lower=0> C_QCD; // correlation matrix for QCD
  array[dmjj-1, dncluster-1,dmjj-1, dncluster-1] real<lower=0> C_top; // correlation matrix for top
}

transformed data {
// no transformed data.
}
parameters {    
  simplex[dncluster-1] theta_ncluster_QCD; // multinomial parameters for QCD ncluster distr.
  simplex[dncluster-1] theta_ncluster_top; // multinomial parameters for QCD ncluster distr.
  simplex[dmjj-1] theta_mjj_QCD; // multinomial parameters for mjj ncluster distr.
  simplex[dmjj-1] theta_mjj_top; // multinomial parameters for mjj ncluster distr.
  simplex[2] class_fractions; // mixture coefficients for QCD and top
}
model {
  class_fractions ~ dirichlet(eta_class_fractions);
  theta_ncluster_QCD ~ dirichlet(eta_ncluster_QCD);  
  theta_ncluster_top ~ dirichlet(eta_ncluster_top); 
  theta_mjj_QCD ~ dirichlet(eta_mjj_QCD);  
  theta_mjj_top ~ dirichlet(eta_mjj_top);
  
  matrix[dmjj-1, dncluster-1] convoluted_QCD;
  matrix[dmjj-1, dncluster-1] convoluted_top;
  
  for (mjj_bin in 1:dmjj-1)
      {
      for (ncluster_bin in 1:dncluster-1)
          {
          convoluted_QCD[mjj_bin,ncluster_bin]=0;
          convoluted_top[mjj_bin,ncluster_bin]=0;
          for (mjj_bin_bis in 1:dmjj-1)
              {
              for (ncluster_bin_bis in 1:dncluster-1)
                  {
                  convoluted_QCD[mjj_bin,ncluster_bin]+=C_QCD[mjj_bin,ncluster_bin,mjj_bin_bis,ncluster_bin_bis]*theta_mjj_QCD[mjj_bin_bis]*theta_ncluster_QCD[ncluster_bin_bis];
                  convoluted_top[mjj_bin,ncluster_bin]+=C_top[mjj_bin,ncluster_bin,mjj_bin_bis,ncluster_bin_bis]*theta_mjj_top[mjj_bin_bis]*theta_ncluster_top[ncluster_bin_bis];
                  }
              }
          }
      }
  
  vector[2] lp;
  for (Ni in 1:N1)     
     {

     lp[1] = log(convoluted_QCD)[mjj1[Ni],ncluster1[Ni]];
     lp[2] = log(convoluted_top)[mjj1[Ni],ncluster1[Ni]];
 
     target += log_mix(class_fractions, lp);
     }
}
     
"""
#
#
mydata_conditional_independence = {'dncluster': bins_ncluster+1,'dmjj': bins_mjj,
                                   'N1': 100000, 'ncluster1': list([int(_) for _ in ncluster1[:100000]]),
                                   'mjj1': list([int(_) for _ in mjj1[:100000]]),
                                   'eta_class_fractions': list(eta_class_fractions),
                                   'eta_ncluster_QCD': list(eta_ncluster_QCD),
                                   'eta_ncluster_top': list(eta_ncluster_top),
                                   'eta_mjj_QCD': list(eta_mjj_QCD),
                                   'eta_mjj_top': list(eta_mjj_top),
                                   'C_QCD': list(Cabcd_QCD),
                                   'C_top': list(Cabcd_top)
                                  }
#
#
posterior = stan.build(model_conditional_independence, data=mydata_conditional_independence, random_seed=20)
#
fit = posterior.sample(num_chains=5, num_samples=300)
#
df = fit.to_frame() 
#
sorted_posterior = df.sort_values(by='lp__', ascending=False)
#
# top fraction figure
#
plt.figure(dpi=140)
plt.hist(sorted_posterior['class_fractions.2'], density=1, bins=np.linspace(0,1,40), label='top')
plt.axvline(true_class_fractions[1], linestyle='dashed', color='black', label='top true')
plt.xlabel('$\pi_1$',fontsize=25)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('frac_con_vi.pdf')
plt.close()
#
print("top fraction mean = ",np.mean(sorted_posterior['class_fractions.2']))
print("top fraction std = ",np.std(sorted_posterior['class_fractions.2']))
#
#
#
M1 = 1500
#
myalphas = np.zeros((2,bins_ncluster,M1))
alpha_mean = np.zeros((2,bins_ncluster))
alpha_err = np.zeros((2,bins_ncluster))
mybetas = np.zeros((2,bins_mjj-1,M1))
beta_mean = np.zeros((2,bins_mjj-1))
beta_err = np.zeros((2,bins_mjj-1))
#
midbins=[[_+0.5 for _ in range(bins_ncluster)],[_+0.5 for _ in range(bins_mjj-1)]]
#
plt.figure(dpi=140)
tot_posterior0 = []
tot_posterior1 = []
for k in range(0,1500):
    
    posterior0=[]
    l = k  
    for i in range(bins_ncluster):
        name = list(sorted_posterior)[7+i]
        posterior0.append(sorted_posterior.iloc[l][name])
    posterior1=[]
    for i in range(bins_ncluster,2*(bins_ncluster)):
        name = list(sorted_posterior)[7+i]
        posterior1.append(sorted_posterior.iloc[k][name])
    tot_posterior0.append(posterior0)
    tot_posterior1.append(posterior1)
#
tot_posterior0 = np.array(tot_posterior0)
tot_posterior1 = np.array(tot_posterior1)
myalphas[0] = tot_posterior0.T
myalphas[1] = tot_posterior1.T
#
alpha_mean[0]=np.mean(myalphas[0],axis=1)
alpha_mean[1]=np.mean(myalphas[1],axis=1)
alpha_err[0]=np.std(myalphas[0],axis=1)
alpha_err[1]=np.std(myalphas[1],axis=1)
#    
plt.hist(midbins[0],bins=np.arange(0,bins_ncluster+1,1),weights=true_QCD_ncluster, histtype='step',linestyle=':', color = 'red')
plt.axhline(1.0, linestyle=':', color='red', label='true QCD')
#           
plt.hist(midbins[0],bins=np.arange(0,bins_ncluster+1,1),weights=true_top_ncluster, histtype='step',linestyle=':', color = 'blue')
plt.axhline(1.0, linestyle=':', color='blue', label='true top')
#
plt.hist(midbins[0],bins=np.arange(0,bins_ncluster+1,1),weights=alpha_mean[0],linestyle='solid', histtype='step',color = 'red')
plt.axhline(1.0, linestyle='solid', color='red', label='mean posterior QCD')
plt.hist(midbins[0],bins=np.arange(0,bins_ncluster+1,1),weights=alpha_mean[1],linestyle='solid', histtype='step',color = 'blue')
plt.axhline(1.0, linestyle='solid', color='blue', label='mean posterior top')
#
nposterior, bposterior, pposterior = plt.hist(midbins[0],bins=np.arange(0,bins_ncluster+1,1),weights=alpha_mean[0],  histtype='step',linestyle='None', color = 'red')
nposterior_up, bposterior_up, pposterior_up = plt.hist(midbins[0],bins=np.arange(0,bins_ncluster+1,1),weights=alpha_mean[0]+alpha_err[0],  histtype='step',linestyle='None',color = 'red')
nposterior_down, bposterior_down, posterior_down = plt.hist(midbins[0],bins=np.arange(0,bins_ncluster+1,1),weights=alpha_mean[0]-alpha_err[0],  histtype='step',linestyle='None', color = 'red')
plt.bar(x=bposterior_up[:-1], height=nposterior_up-nposterior_down, bottom=nposterior_down, width=np.diff(bposterior_up), align='edge', linewidth=1, color='None', edgecolor='red',alpha=0.5, zorder=-1, hatch = "\ \ ",label='posterior QCD')
#
nposterior1, bposterior1, pposterior1 = plt.hist(midbins[0],bins=np.arange(0,bins_ncluster+1,1),weights=alpha_mean[1],  histtype='step',linestyle='None', color = 'blue')
nposterior_up1, bposterior_up1, pposterior_up1 = plt.hist(midbins[0],bins=np.arange(0,bins_ncluster+1,1),weights=alpha_mean[1]+alpha_err[1],  histtype='step',linestyle='None', color = 'blue')
nposterior_down1, bposterior_down1, pposterior_down1 = plt.hist(midbins[0],bins=np.arange(0,bins_ncluster+1,1),weights=alpha_mean[1]-alpha_err[1],  histtype='step',linestyle='None', color = 'blue')
plt.bar(x=bposterior_up1[:-1], height=nposterior_up1-nposterior_down1, bottom=nposterior_down1, width=np.diff(bposterior_up1), align='edge', linewidth=1, edgecolor='blue',color='None', alpha=0.5, zorder=-1, hatch = '//',label='posteror top')
#
nprior, bprior, pprior = plt.hist(midbins[0],bins=np.arange(0,bins_ncluster+1,1),weights=eta_ncluster_QCD/np.sum(eta_ncluster_QCD),  histtype='step',linestyle='None', color = 'red')
nprior_up, bprior_up, pprior_up = plt.hist(midbins[0],bins=np.arange(0,bins_ncluster+1,1),weights=eta_ncluster_QCD/np.sum(eta_ncluster_QCD) + np.sqrt(dirichlet.var(alpha=eta_ncluster_QCD)),  histtype='step',linestyle='None', color = 'red')
nprior_down, bprior_down, pprior_down = plt.hist(midbins[0],bins=np.arange(0,bins_ncluster+1,1),weights=eta_ncluster_QCD/np.sum(eta_ncluster_QCD) - np.sqrt(dirichlet.var(alpha=eta_ncluster_QCD)),  histtype='step',linestyle='None', color = 'red')
plt.bar(x=bprior_up[:-1], height=nprior_up-nprior_down, bottom=nprior_down, width=np.diff(bprior_up), align='edge', linewidth=0, color='red', alpha=0.25, zorder=-1,label='prior QCD')
#
nprior1, bprior1, pprior1 = plt.hist(midbins[0],bins=np.arange(0,bins_ncluster+1,1),weights=eta_ncluster_top/np.sum(eta_ncluster_top),  histtype='step',linestyle='None', color = 'blue')
nprior_up1, bprior_up1, pprior_up1 = plt.hist(midbins[0],bins=np.arange(0,bins_ncluster+1,1),weights=eta_ncluster_top/np.sum(eta_ncluster_top) + np.sqrt(dirichlet.var(alpha=eta_ncluster_top)),  histtype='step',linestyle='None', color = 'blue')
nprior_down1, bprior_down1, pprior_down1 = plt.hist(midbins[0],bins=np.arange(0,bins_ncluster+1,1),weights=eta_ncluster_top/np.sum(eta_ncluster_top) - np.sqrt(dirichlet.var(alpha=eta_ncluster_top)),  histtype='step',linestyle='None', color = 'blue')
plt.bar(x=bprior_up1[:-1], height=nprior_up1-nprior_down1, bottom=nprior_down1, width=np.diff(bprior_up1), align='edge', linewidth=0, color='blue', alpha=0.25, zorder=-1,label='prior top')
plt.ylim([0.,0.58])
#
plt.xlabel('$N_\mathrm{clus}$',fontsize=25)
plt.xticks(midbins[0],[2+i for i in range(bins_ncluster)])
plt.tight_layout()
plt.savefig('clus_con_vi.pdf')
plt.close()
#
plt.figure(dpi=140)
tot_posterior0 = []
tot_posterior1 = []
for k in range(0,1500):
    posterior0=[]
    l = np.random.randint(len(sorted_posterior)) 
    for i in range(bins_mjj-1):
        name = list(sorted_posterior)[7+2*(bins_ncluster)+i]
        posterior0.append(sorted_posterior.iloc[l][name])
    posterior1=[]
    for i in range(bins_mjj-1,2*(bins_mjj-1)):
        name = list(sorted_posterior)[7+2*(bins_ncluster)+i]
        posterior1.append(sorted_posterior.iloc[k][name])
    tot_posterior0.append(posterior0)
    tot_posterior1.append(posterior1)
tot_posterior0 = np.array(tot_posterior0)
tot_posterior1 = np.array(tot_posterior1)
mybetas[0] = tot_posterior0.T
mybetas[1] = tot_posterior1.T
#
beta_mean[0]=np.mean(mybetas[0],axis=1)
beta_mean[1]=np.mean(mybetas[1],axis=1)
beta_err[0]=np.std(mybetas[0],axis=1)
beta_err[1]=np.std(mybetas[1],axis=1)
#
plt.hist(midbins[1],bins=np.arange(0,bins_mjj,1),weights=true_QCD_mjj, histtype='step',linestyle=':', color = 'red')
plt.axhline(1.0, linestyle=':', color='red', label='true QCD')
#
plt.hist(midbins[1],bins=np.arange(0,bins_mjj,1),weights=true_top_mjj, histtype='step',linestyle=':', color = 'blue')
plt.axhline(1.0, linestyle=':', color='blue', label='true top')
#
plt.hist(midbins[1],bins=np.arange(0,bins_mjj,1),weights=beta_mean[0],linestyle='solid', histtype='step',color = 'red')
plt.axhline(1.0, linestyle='solid', color='red', label='mean posterior QCD')

plt.hist(midbins[1],bins=np.arange(0,bins_mjj,1),weights=beta_mean[1],linestyle='solid', histtype='step',color = 'blue')
plt.axhline(1.0, linestyle='solid', color='blue', label='mean posterior top')
#
nposterior, bposterior, pposterior = plt.hist(midbins[1],bins=np.arange(0,bins_mjj,1),weights=beta_mean[0],  histtype='step',linestyle='None', color = 'red')
nposterior_up, bposterior_up, pposterior_up = plt.hist(midbins[1],bins=np.arange(0,bins_mjj,1),weights=beta_mean[0]+beta_err[0],  histtype='step',linestyle='None',color = 'red')
nposterior_down, bposterior_down, posterior_down = plt.hist(midbins[1],bins=np.arange(0,bins_mjj,1),weights=beta_mean[0]-beta_err[0],  histtype='step',linestyle='None', color = 'red')
plt.bar(x=bposterior_up[:-1], height=nposterior_up-nposterior_down, bottom=nposterior_down, width=np.diff(bposterior_up), align='edge', linewidth=1, color='None', edgecolor='red',alpha=0.5, zorder=-1, hatch = "\ \ ",label='posterior QCD')
#
nposterior1, bposterior1, pposterior1 = plt.hist(midbins[1],bins=np.arange(0,bins_mjj,1),weights=beta_mean[1],  histtype='step',linestyle='None', color = 'blue')
nposterior_up1, bposterior_up1, pposterior_up1 = plt.hist(midbins[1],bins=np.arange(0,bins_mjj,1),weights=beta_mean[1]+beta_err[1],  histtype='step',linestyle='None', color = 'blue')
nposterior_down1, bposterior_down1, pposterior_down1 = plt.hist(midbins[1],bins=np.arange(0,bins_mjj,1),weights=beta_mean[1]-beta_err[1],  histtype='step',linestyle='None', color = 'blue')
plt.bar(x=bposterior_up1[:-1], height=nposterior_up1-nposterior_down1, bottom=nposterior_down1, width=np.diff(bposterior_up1), align='edge', linewidth=1, edgecolor='blue',color='None', alpha=0.5, zorder=-1, hatch = '//',label='posteror top')
#
nprior, bprior, pprior = plt.hist(midbins[1],bins=np.arange(0,bins_mjj,1),weights=eta_mjj_QCD/np.sum(eta_mjj_QCD),  histtype='step',linestyle='None', color = 'red')
nprior_up, bprior_up, pprior_up = plt.hist(midbins[1],bins=np.arange(0,bins_mjj,1),weights=eta_mjj_QCD/np.sum(eta_mjj_QCD) + np.sqrt(dirichlet.var(alpha=eta_mjj_QCD)),  histtype='step',linestyle='None', color = 'red')
nprior_down, bprior_down, pprior_down = plt.hist(midbins[1],bins=np.arange(0,bins_mjj,1),weights=eta_mjj_QCD/np.sum(eta_mjj_QCD) - np.sqrt(dirichlet.var(alpha=eta_mjj_QCD)),  histtype='step',linestyle='None', color = 'red')
plt.bar(x=bprior_up[:-1], height=nprior_up-nprior_down, bottom=nprior_down, width=np.diff(bprior_up), align='edge', linewidth=0, color='red', alpha=0.25, zorder=-1,label='prior QCD')
#
nprior1, bprior1, pprior1 = plt.hist(midbins[1],bins=np.arange(0,bins_mjj,1),weights=eta_mjj_top/np.sum(eta_mjj_top),  histtype='step',linestyle='None', color = 'blue')
nprior_up1, bprior_up1, pprior_up1 = plt.hist(midbins[1],bins=np.arange(0,bins_mjj,1),weights=eta_mjj_top/np.sum(eta_mjj_top) + np.sqrt(dirichlet.var(alpha=eta_mjj_top)),  histtype='step',linestyle='None', color = 'blue')
nprior_down1, bprior_down1, pprior_down1 = plt.hist(midbins[1],bins=np.arange(0,bins_mjj,1),weights=eta_mjj_top/np.sum(eta_mjj_top) - np.sqrt(dirichlet.var(alpha=eta_mjj_top)),  histtype='step',linestyle='None', color = 'blue')
plt.bar(x=bprior_up1[:-1], height=nprior_up1-nprior_down1, bottom=nprior_down1, width=np.diff(bprior_up1), align='edge', linewidth=0, color='blue', alpha=0.25, zorder=-1,label='prior top')
#
linevalue = eta_mjj_top[3]/np.sum(eta_mjj_top)
linearray = np.array([linevalue,linevalue,linevalue,linevalue,linevalue])
plt.plot([3,4,5,6,7],linearray,color='cyan')
plt.ylim([0.,0.180])
plt.xlabel(r'$\mathrm{Mass\, [GeV]}$',fontsize=25)
plt.xticks([0,2,4,6,8,10,12,14],[150,160,170,180,190,200,210,220])
plt.tight_layout()
plt.savefig('mass_con_vi.pdf')
plt.close()
#
del M1
del myalphas
del mybetas
del alpha_mean
del beta_mean
del alpha_err
del beta_err
#
# calculating the distances
#
#The number 750 is because we have 1500 posterior samples in total
M1 = 750
M1p = 750
#
myalphas = np.zeros((2,bins_ncluster,M1))
mybetas = np.zeros((2,bins_mjj-1,M1))
mypies  = np.zeros((2,M1))
#
myalphas2 = np.zeros((2,bins_ncluster,M1p))
mybetas2 = np.zeros((2,bins_mjj-1,M1p))
mypies2  = np.zeros((2,M1p))
#
myalphas_mean = np.zeros((2,bins_ncluster))
mybetas_mean = np.zeros((2,bins_mjj-1))
mypies_mean  = np.zeros((2))
#
myalphas2_mean = np.zeros((2,bins_ncluster))
mybetas2_mean = np.zeros((2,bins_mjj-1))
mypies2_mean  = np.zeros((2))
#
myalphas_std = np.zeros((2,bins_ncluster))
mybetas_std = np.zeros((2,bins_mjj-1))
mypies_std  = np.zeros((2))
#
myalphas2_std = np.zeros((2,bins_ncluster))
mybetas2_std = np.zeros((2,bins_mjj-1))
mypies2_std  = np.zeros((2))
#
for num in range(1,bins_ncluster+1):
    myalphas[0][num-1] = sorted_posterior['theta_ncluster_QCD.{0}'.format(num)][:M1]
    myalphas[1][num-1] = sorted_posterior['theta_ncluster_top.{0}'.format(num)][:M1]
    myalphas_mean[0][num-1] = np.mean(sorted_posterior['theta_ncluster_QCD.{0}'.format(num)][:M1])
    myalphas_mean[1][num-1] = np.mean(sorted_posterior['theta_ncluster_top.{0}'.format(num)][:M1])
    myalphas_std[0][num-1] = np.std(sorted_posterior['theta_ncluster_QCD.{0}'.format(num)][:M1])
    myalphas_std[1][num-1] = np.std(sorted_posterior['theta_ncluster_top.{0}'.format(num)][:M1])
#  
for num in range(1,bins_mjj):
    mybetas[0][num-1] = sorted_posterior['theta_mjj_QCD.{0}'.format(num)][:M1]
    mybetas[1][num-1] = sorted_posterior['theta_mjj_top.{0}'.format(num)][:M1]
    mybetas_mean[0][num-1] = np.mean(sorted_posterior['theta_mjj_QCD.{0}'.format(num)][:M1])
    mybetas_mean[1][num-1] = np.mean(sorted_posterior['theta_mjj_top.{0}'.format(num)][:M1])
    mybetas_std[0][num-1] = np.std(sorted_posterior['theta_mjj_QCD.{0}'.format(num)][:M1])
    mybetas_std[1][num-1] = np.std(sorted_posterior['theta_mjj_top.{0}'.format(num)][:M1])  
#   
mypies[0] = sorted_posterior['class_fractions.1'][:M1]
mypies[1] = sorted_posterior['class_fractions.2'][:M1]
mypies_mean[0] = np.mean(sorted_posterior['class_fractions.1'][:M1])
mypies_mean[1] = np.mean(sorted_posterior['class_fractions.2'][:M1])
mypies_std[0] = np.std(sorted_posterior['class_fractions.1'][:M1])
mypies_std[1] = np.std(sorted_posterior['class_fractions.2'][:M1])
#
for n1 in range(M1p):
    myalphas2[0,:,n1] = stats.dirichlet(alpha=eta_ncluster_QCD).rvs()[0]
    myalphas2[1,:,n1] = stats.dirichlet(alpha=eta_ncluster_top).rvs()[0]
    mybetas2[0,:,n1] = stats.dirichlet(alpha=eta_mjj_QCD).rvs()[0]
    mybetas2[1,:,n1] = stats.dirichlet(alpha=eta_mjj_top).rvs()[0]
    mypies2[:,n1] = stats.dirichlet(alpha=eta_class_fractions).rvs()[0]
#
for num in range(1,bins_ncluster+1):
    myalphas2_mean[0][num-1] = np.mean(myalphas2[0][num-1][:])
    myalphas2_mean[1][num-1] = np.mean(myalphas2[1][num-1][:])
    myalphas2_std[0][num-1] = np.std(myalphas2[0][num-1][:])
    myalphas2_std[1][num-1] = np.std(myalphas2[1][num-1][:])    
#  
for num in range(1,bins_mjj):
    mybetas2_mean[0][num-1] = np.mean(mybetas2[0][num-1][:])
    mybetas2_mean[1][num-1] = np.mean(mybetas2[1][num-1][:])
    mybetas2_std[0][num-1] = np.std(mybetas2[0][num-1][:])
    mybetas2_std[1][num-1] = np.std(mybetas2[1][num-1][:])    
mypies2_mean[0] = np.mean(mypies2[0][:])
mypies2_mean[1] = np.mean(mypies2[1][:])
mypies2_std[0] = np.std(mypies2[0][:])
mypies2_std[1] = np.std(mypies2[1][:])
#
#calculating the distance for every topic
#
dist_alpha_post = np.zeros((4,bins_ncluster))
dist_alpha_prior = np.zeros((4,bins_ncluster))
dist_beta_post = np.zeros((4,bins_mjj-1))
dist_beta_prior = np.zeros((4,bins_mjj-1))
dist_pi_post  = np.zeros((4))
dist_pi_prior  = np.zeros((4))
#
for num in range(1,bins_ncluster+1):
    dist_alpha_post[0][num-1] = np.absolute(myalphas_mean[0][num-1] - true_QCD_ncluster[num-1])
    dist_alpha_post[1][num-1] = np.absolute(myalphas_mean[1][num-1] - true_top_ncluster[num-1])
    dist_alpha_post[2][num-1] = np.absolute(myalphas_std[0][num-1])
    dist_alpha_post[3][num-1] = np.absolute(myalphas_std[1][num-1])
    dist_alpha_prior[0][num-1] = np.absolute(myalphas2_mean[0][num-1] - true_QCD_ncluster[num-1])
    dist_alpha_prior[1][num-1] = np.absolute(myalphas2_mean[1][num-1] - true_top_ncluster[num-1])
    dist_alpha_prior[2][num-1] = np.absolute(myalphas2_std[0][num-1])
    dist_alpha_prior[3][num-1] = np.absolute(myalphas2_std[1][num-1])
#    
for num in range(1,bins_mjj):
    dist_beta_post[0][num-1] = np.absolute(mybetas_mean[0][num-1] - true_QCD_mjj[num-1])
    dist_beta_post[1][num-1] = np.absolute(mybetas_mean[1][num-1] - true_top_mjj[num-1])
    dist_beta_post[2][num-1] = np.absolute(mybetas_std[0][num-1])
    dist_beta_post[3][num-1] = np.absolute(mybetas_std[1][num-1])
    dist_beta_prior[0][num-1] = np.absolute(mybetas2_mean[0][num-1] - true_QCD_mjj[num-1])
    dist_beta_prior[1][num-1] = np.absolute(mybetas2_mean[1][num-1] - true_top_mjj[num-1])
    dist_beta_prior[2][num-1] = np.absolute(mybetas2_std[0][num-1])
    dist_beta_prior[3][num-1] = np.absolute(mybetas2_std[1][num-1])    
dist_pi_post[0] = np.absolute(mypies_mean[0] - true_class_fractions[0])
dist_pi_post[1] = np.absolute(mypies_mean[1] - true_class_fractions[1])
dist_pi_post[2] = np.absolute(mypies_std[0])
dist_pi_post[3] = np.absolute(mypies_std[1])
#
dist_pi_prior[0] = np.absolute(mypies2_mean[0] - true_class_fractions[0])
dist_pi_prior[1] = np.absolute(mypies2_mean[1] - true_class_fractions[1])
dist_pi_prior[2] = np.absolute(mypies2_std[0])
dist_pi_prior[3] = np.absolute(mypies2_std[1])
#
# saving the data to the files
#
np.save('dist_alpha_post.npy',dist_alpha_post)
np.save('dist_alpha_prior.npy',dist_alpha_prior)
np.save('dist_beta_post.npy',dist_beta_post)
np.save('dist_beta_prior.npy',dist_beta_prior)
np.save('dist_pi_post.npy',dist_pi_post)
np.save('dist_pi_prior.npy',dist_pi_prior)
#########################################
#
# MAPs calculations
#
##########################################
#############################################################

pAB_true = np.zeros((2,bins_mjj-1,bins_ncluster))
mypAB = np.zeros((2,bins_mjj-1,bins_ncluster,M1))
#
for mjj_bin in range(bins_mjj-1):
    for ncluster_bin in range(bins_ncluster):
        for mjj_bin_bis in range(bins_mjj-1):
            for ncluster_bin_bis in range(bins_ncluster):
                mypAB[0,mjj_bin,ncluster_bin]+=Cabcd_QCD[mjj_bin,ncluster_bin,mjj_bin_bis,ncluster_bin_bis]*myalphas[0,ncluster_bin_bis]*mybetas[0,mjj_bin_bis]
                mypAB[1,mjj_bin,ncluster_bin]+=Cabcd_top[mjj_bin,ncluster_bin,mjj_bin_bis,ncluster_bin_bis]*myalphas[1,ncluster_bin_bis]*mybetas[1,mjj_bin_bis]
                mypAB2[0,mjj_bin,ncluster_bin]+=Cabcd_QCD[mjj_bin,ncluster_bin,mjj_bin_bis,ncluster_bin_bis]*myalphas2[0,ncluster_bin_bis]*mybetas2[0,mjj_bin_bis]
                mypAB2[1,mjj_bin,ncluster_bin]+=Cabcd_top[mjj_bin,ncluster_bin,mjj_bin_bis,ncluster_bin_bis]*myalphas2[1,ncluster_bin_bis]*mybetas2[1,mjj_bin_bis]
                pAB_true[0,mjj_bin,ncluster_bin]+=Cabcd_QCD[mjj_bin,ncluster_bin,mjj_bin_bis,ncluster_bin_bis]*true_QCD_ncluster[ncluster_bin_bis]*true_QCD_mjj[mjj_bin_bis]
                pAB_true[1,mjj_bin,ncluster_bin]+=Cabcd_top[mjj_bin,ncluster_bin,mjj_bin_bis,ncluster_bin_bis]*true_top_ncluster[ncluster_bin_bis]*true_top_mjj[mjj_bin_bis]
pies, pAB = np.mean(mypies,axis=-1),np.mean(mypAB,axis=-1)
pies2, pAB2 = np.mean(mypies2,axis=-1),np.mean(mypAB2,axis=-1)
#
map_qcd_post = np.zeros((bins_ncluster,bins_mjj-1))
map_top_post = np.zeros((bins_ncluster,bins_mjj-1))
map_post =  np.zeros((bins_ncluster,bins_mjj-1))
#
map_qcd_prior = np.zeros((bins_ncluster,bins_mjj-1))
map_top_prior = np.zeros((bins_ncluster,bins_mjj-1))
map_prior =  np.zeros((bins_ncluster,bins_mjj-1))
#
map_qcd_true = np.zeros((bins_ncluster,bins_mjj-1))
map_top_true = np.zeros((bins_ncluster,bins_mjj-1))
#
for ic in range(0,5):
    for im in range(0,15):
        mysum_post = pAB[0,im,ic] + pAB[1,im,ic]
        map_qcd_post[ic,im]= pAB[0,im,ic]
        map_top_post[ic,im]= pAB[1,im,ic]
        map_post[ic,im] = np.mean(mypies,axis=-1)[0]*pAB[0,im,ic] + np.mean(mypies,axis=-1)[1]*pAB[1,im,ic]
#
        map_qcd_prior[ic,im] = pAB2[0,im,ic]
        map_top_prior[ic,im] = pAB2[1,im,ic]
        map_prior[ic,im] = np.mean(mypies2,axis=-1)[0]*pAB2[0,im,ic] + np.mean(mypies2,axis=-1)[1]*pAB2[1,im,ic]
#        
        mysum_true = pAB_true[0,im,ic] + pAB_true[1,im,ic]
        map_qcd_true[ic,im]= pAB_true[0,im,ic]
        map_top_true[ic,im]= pAB_true[1,im,ic]
#
fig = plt.figure()
ax = fig.add_subplot(111)          
pos0 = ax.imshow(map_top_post.T[:,:],origin='lower',aspect=0.3,vmin=0, vmax=0.1,cmap='gist_heat_r'
                )
ax.xaxis.set(ticks=[0,1,2,3,4],ticklabels=[2,3,4,5,6])
ax.yaxis.set(ticks=np.arange(-0.5,15,2),ticklabels=np.arange(150,225,10))
ax.tick_params(axis='both', which='major', labelsize=18)
ax.set_xlabel(r'$N_{\mathrm{clus}}$', fontsize=22)
ax.set_ylabel(r'$\mathrm{Mass\, [GeV]}$',fontsize=22)
cbar=fig.colorbar(pos1,ax=ax,fraction=0.15,location='right',shrink=1)
cbar.ax.tick_params(labelsize=12) 
ax.set_title("$\Sigma=1400$ posterior",size=23)
fig.tight_layout(pad=0.15)
fig.savefig('map_con_vi.pdf')
plt.close()
#
fig = plt.figure()
ax = fig.add_subplot(111)         
pos1 = ax.imshow(map_top_true.T[:,:],origin='lower',aspect=0.3,vmin=0, vmax=0.1,cmap='gist_heat_r'
                    )
ax.xaxis.set(ticks=[0,1,2,3,4],ticklabels=[2,3,4,5,6])
ax.yaxis.set(ticks=np.arange(-0.5,15,2),ticklabels=np.arange(150,225,10))
ax.tick_params(axis='both', which='major', labelsize=18)
ax.set_xlabel(r'$N_{\mathrm{clus}}$', fontsize=22)
ax.set_ylabel(r'$\mathrm{Mass\, [GeV]}$',fontsize=22)
cbar=fig.colorbar(pos1,ax=ax,fraction=0.15,location='right',shrink=1)
cbar.ax.tick_params(labelsize=12)
ax.set_title("true",size=23)
fig.tight_layout(pad=0.15,h_pad=0, w_pad=0, rect=None)
fig.savefig('map_true.pdf')
plt.close()
#############################################################
#############################################################
########################################################################
#
# KL divergence
#
########################################################################
map_qcd_post_kl = map_qcd_post.T
map_top_post_kl = map_top_post.T
map_post_kl = map_post.T
#
map_qcd_prior_kl = map_qcd_prior.T
map_top_prior_kl = map_top_prior.T
map_prior_kl = map_prior.T
#
kl_post_qcd = np.zeros((5,15))
kl_post_top = np.zeros((5,15))
kl_post = np.zeros((5,15))
#
kl_prior_qcd = np.zeros((5,15))
kl_prior_top = np.zeros((5,15))
kl_prior = np.zeros((5,15))
#
NAB_QCD_kl = NAB_QCD_kl/np.sum(NAB_QCD_kl)
NAB_top_kl = NAB_top_kl/np.sum(NAB_top_kl)
NAB_kl = NAB_kl/np.sum(NAB_kl)
#
#print("test2 = ",np.sum(NAB_kl))
#print("test3 = ",np.sum(map_post))
#
for ic in range(0,5):
    for im in range(0,15):
        kl_post_qcd[ic,im] = map_qcd_post[ic,im]*np.log(map_qcd_post[ic,im]/NAB_QCD_kl[im,ic])
        kl_post_top[ic,im] = map_top_post[ic,im]*np.log(map_top_post[ic,im]/NAB_top_kl[im,ic])
        kl_post[ic,im] = map_post[ic,im]*np.log(map_post[ic,im]/NAB_kl[im,ic])
        #
        kl_prior_qcd[ic,im] = map_qcd_prior[ic,im]*np.log(map_qcd_prior[ic,im]/NAB_QCD_kl[im,ic])
        kl_prior_top[ic,im] = map_top_prior[ic,im]*np.log(map_top_prior[ic,im]/NAB_top_kl[im,ic])
        kl_prior[ic,im] = map_prior[ic,im]*np.log(map_prior[ic,im]/NAB_kl[im,ic])
#
print("my KL div post QCD = ",np.sum(kl_post_qcd))
print("my KL div post top = ",np.sum(kl_post_top))
print("my KL div post total = ",np.sum(kl_post))
print("my KL div prior QCD = ",np.sum(kl_prior_qcd))
print("my KL div prior top = ",np.sum(kl_prior_top))
print("my KL div pror total = ",np.sum(kl_prior))
#
