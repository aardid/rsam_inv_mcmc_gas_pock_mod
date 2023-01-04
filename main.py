#############################################################################
import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd 
import math, random
import matplotlib.pyplot as plt
import emcee, corner
import multiprocessing 
import time
import os
# define constant
pi=np.pi
sqrt=np.sqrt
##############################################################################
def _load_rsam_data(ti,tf, downsample=None, fl=None):
    '''
    Returns a data frame of RSAM between dates ti and tf (default from Ruapehu data)
    i.e.,
    ti='2016-11-13 12:00:00'
    tf='2016-11-28'
    downsample: value given correspond to the step size. For example, if the original sampling is
    every 10 mins, a step size of 6 will resample to every hour
    '''
    # import RSAM Ruapehu data
    if fl is None:
        fl= 'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\volc_forecast_tl\\data\\FWVZ_tremor_data.csv'
    df = pd.read_csv(fl, index_col= 'time', usecols=['time','rsamF'], infer_datetime_format=True)
    # Filter between dates 
    df = df.loc[ti:tf]
    # Converting the index as date
    df.index = pd.to_datetime(df.index)
    if downsample:
        step_size=downsample
        df = df.iloc[::step_size, :]
    return df
def _4mcmc_fwd_cav_Ps_Uz_Vz(D=None,Q=None):
    '''
    Forward function from Girona et al(2019) to compute DP, DP2, U_z, V_z
    Q  #mean gas flux
    D   #gas pocket thickness

    Defaults:
    tau=600;                                                        #seconds of simulation
    N=1;                                                            #number of mass impulses in tau seconds
    R=200;                                                          #conduit radius
    L=50;                                                           #thickness of the cap
    distance=2000;                                                  #source-receiver distance
    max_freq=50;                                                    #maximum frequency to be reached in the simulation
    '''
    #D=0.01;                                                         #gas pocket thickness
    #Q=50;                                                           #mean gas flux

    tau=30;#600                                                        #seconds of simulation
    N=1000;#1                                                            #number of mass impulses in tau seconds
    R=200;                                                          #conduit radius
    L=50;                                                           #thickness of the cap
    distance=2000;                                                  #source-recSeiver distance
    max_freq=50;                                                    #maximum frequency to be reached in the simulation
    
    ## PHYSICAL PARAMETERS
    S=pi*R**2;                                                                  #conduit section
    mu_g=1e-5;                                                                  #gas viscosity
    T=1000+273.15;                                                              #gas temperature
    M=0.018;                                                                    #molecular weight of gas (water vapor)
    Rg=8.3145;                                                                  #ideal gas constant
    Pex=101325;                                                                 #external pressure
    kappa=1e-8;                                                                 #permeability of the cap
    phi=0.0001;                                                                 #porosity of the cap
    rho_s=3000;                                                                 #density of the medium of propagation
    Qf=20;                                                                      #quality factor. 
    ## AUXILIARY PARAMETERS
    beta_a=S*phi*M/(Rg*T);
    beta_b=mu_g*phi/(kappa*(Pex-Rg*T*Q**2/(S**2*phi**2*M*Pex)));
    beta_c=Pex*M/(Rg*T*(Pex-Rg*T*Q**2/(S**2*phi**2*M*Pex)));
    beta_d=2*Q/(S*phi*(Pex-Rg*T*Q**2/(S**2*phi**2*M*Pex)));
    beta_e=S*M*D/(Rg*T);
    P0=Pex+mu_g*Rg*T*Q*L/(S*kappa*M*(Pex-Rg*T*Q**2/(S**2*phi**2*M*Pex)));
    ## COEFFICIENTS OF THE HARMONIC OSCILLATOR
    GAMMA0=1;
    GAMMA1=(2*(beta_a *beta_d+beta_b *beta_e )*L+beta_a *beta_b *L**2)/(2*beta_a );
    GAMMA2=(2*beta_c *beta_e *L+beta_a *beta_c *L**2)/(2*beta_a );
    gamma0=beta_b *L/beta_a;
    gamma1=beta_c*L/beta_a;
    ## NATURAL FREQUENCY AND CRITICAL THICKNESS
    fn=sqrt((sqrt((GAMMA2*gamma0**2+gamma1**2)**2-GAMMA1**2*gamma0**2*gamma1**2)-GAMMA2*gamma0**2)/(GAMMA2*gamma1**2))/(2*pi);
    a0=(beta_a*beta_b*L**2+2*beta_a*beta_d*L)**2*beta_b**2-4*beta_a**2*beta_b**2*beta_c*L**2-4*beta_a**2*beta_c**2;
    a1=4*(beta_a*beta_b*L**2+2*beta_a*beta_d*L)*beta_b**3*L-8*beta_a*beta_b**2*beta_c*L;
    a2=4*beta_b**4*L**2;
    Dcrit=(Rg*T/(S*M))*(-a1+sqrt(a1**2-4*a0*a2))/(2*a2);
    #print('Natural frequency of the oscillator is: ' +str(fn)+ ' Hz')
    #print('Critical thickness of the gas pocket: '+str(Dcrit)+ ' m')
    ## SIMULATION PARAMETERS
    #tau=50;                                                         #seconds of simulation
    #N=10;                                                           #number of mass impulses in tau seconds
    dt=1/max_freq;                                                  #sampling rate
    inter=1/tau;                                                    #sampling frequency
    t=np.arange(0,tau,dt);                                          #time vector
    ## SUPPLY OF VOLATILES TO THE GAS POCKET  
    t0=np.random.uniform(0,tau,N)
    t0=np.sort(t0);                                               #instant at which impulses occur. Uniformly distributed random times
    qn_ave=Q*tau/N;                                                             #average mass of the bubbles that burst at the top of the magma column
    # (matlab version) qn=qn_ave+0*normrnd(0,20*qn_ave/100,1,N);
    qn=1*np.random.normal(qn_ave, 0.1*qn_ave, N);                               #mass contained in each bubble that bursts. It is normally distributed around qn_ave
    #while length(qn(qn<0))>0;qn=qn_ave+normrnd(0,20*qn_ave/100,1,N);end         #this is to avoid mass impulses with negative mass.    
    ## CALCULATION OF THE PRESSURE EVOLUTION IN THE GAS POCKET IN THE TIME DOMAIN [using equation(15)]
    DP = []
    if False:
        DP_old=0;
        GAMMA=GAMMA1/(2*GAMMA2);
        OMEGA=sqrt(abs((4*GAMMA0*GAMMA2-GAMMA1**2)/(4*GAMMA2**2)));
        aux = np.zeros((len(t),N))
        for k in range(N):
            if 4*GAMMA0*GAMMA2>=GAMMA1**2:
                aux[:,k]=qn[k]*np.heaviside(t-t0[k],0.5)*np.exp(-GAMMA*(t-t0[k]))*(((gamma0-gamma1*GAMMA)/(OMEGA*GAMMA2))*np.sin(OMEGA*(t-t0[k]))+(gamma1/GAMMA2)*np.cos(OMEGA*(t-t0[k])));
            else:
                aux[:,k]=qn[k]*np.heaviside(t-t0[k],0.5)*np.exp(-GAMMA*(t-t0[k]))*(((gamma0-gamma1*GAMMA)/(OMEGA*GAMMA2))*np.sinh(OMEGA*(t-t0[k]))+(gamma1/GAMMA2)*np.cosh(OMEGA*(t-t0[k])));
            DP=DP_old+aux[:,k];
            DP_old=DP;  
        DP=DP+Pex-P0;
    ## CALCULATION OF THE PRESSURE EVOLUTION IN THE GAS POCKET AND GROUND DISPLACEMENT IN THE FREQUENCY DOMAIN [using equation (C11)]
    # note that from the frequency domain we calculate the steady-state pressure
    # evolution, and thus we do not account for the transient state. That
    # is why the pressure time series calculated with this approach does not
    # exactly coincide with the pressure time series calculated directly in the time domain 
    # (equation (15)) during the first seconds of simulation. 
    m=0;
    #i=1j;
    _loop = np.arange(0,1/dt,1/tau)
    freq_teo = 0*_loop#np.zeros(len(_loop)) 
    A_res = 1j*0*_loop
    A_exc = 1j*0*_loop
    A_p = 1j*0*_loop
    A_path = 1j*0*_loop
    u_z = 1j*0*_loop
    v_z = 1j*0*_loop
    #
    for j in _loop:#=0:1/tau:1/dt
        ome=2*pi*j;
        freq_teo[m]=j; 

        #source
        A_res[m]=(gamma0*(1j*ome)**0+gamma1*(1j*ome)**1)/(GAMMA0*(1j*ome)**0+GAMMA1*(1j*ome)**1+GAMMA2*(1j*ome)**2);  
        A_exc[m]=np.sum(qn*np.exp(-1j*ome*t0));                                               
        A_p[m]=A_res[m]*A_exc[m];                                                    

        #pathway
        vc=1295*(ome/(2*pi))**-0.374;        #phase velocity 
        vu=0.73*vc;                         #group velocity
        if ome==0:
            A_path[m]=0;
        else:
            A_path[m]=np.exp(1j*((ome/vc)*distance+pi/4))*(1/(8*rho_s*vc**2*vu))*np.sqrt(2*vc*ome/(pi*distance))*np.exp(-ome*distance/(2*vu*Qf));

        #vertical ground displacement
        u_z[m]=S*A_res[m]*A_exc[m]*A_path[m];     #ground displacement in frequency domain
        v_z[m]=ome*u_z[m]#S*A_ress[m]*A_exc[m]*A_path[m];     #ground displacement in frequency domain

        m=m+1;
    # reconstruction of the signals in the time domain
    #P=ifft(A_p,'symmetric')*max_freq;
    P=np.fft.ifft(A_p)*max_freq;
    DP2=P+Pex-P0;
    #U_z=ifft(u_z,'symmetric')*max_freq;
    U_z=np.fft.ifft(u_z)*max_freq;
    V_z=np.fft.ifft(v_z)*max_freq;
    #
    return DP, DP2, U_z, V_z
## MCMC functions
def prob_likelihood(est, obs):
    v=1#np.std(obs)#1
    norm=2
    #prob = -.5*(np.sum((est - obs)**norm))/(v**norm)
    prob= -.5* (np.linalg.norm(obs-est)/(v**norm))
    return prob
def lnprob(pars, obs):
    ## Parameter constraints
    if (any(x<0 for x in pars)): # positive parameters
        return -np.Inf
    if pars[0] > .1:# max(self.meb_prof)+5.: # percentage range 
        return -np.Inf
    if pars[1] > 100:# max(self.meb_prof)+5.: # percentage range 
        return -np.Inf

    ## estimate square function of clay content given pars [z1,z2,%] 
    DP, DP2, U_z, V_z = _4mcmc_fwd_cav_Ps_Uz_Vz(pars[0],int(pars[1]))
    est = np.mean(np.abs(V_z)*1e6)

    ## calculate prob and return  
    prob = prob_likelihood(est,obs)
    ## check if prob is nan
    if prob!=prob: # assign non values to -inf
        return -np.Inf
    return prob 
def run_mcmc(data_p, walk_jump = None):
    '''
    run mcmc. Results are save in local directory in chain.dat file
    data_p: rsam value to fit 
    '''
    nwalkers= 10        # number of walkers
    ndim = 2               # parameter space dimensionality
    if walk_jump is None:
        walk_jump = 200
    ## Timing inversion
    #start_time = time.time()
    # create the emcee object (set threads>1 for multiprocessing)
    data = data_p#df['rsam'][0]# U_z_noise
    cores = multiprocessing.cpu_count()
    # Create sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[data], threads=cores-1)
    # set the initial location of the walkers
    _p0_D=np.random.uniform(0,.1,nwalkers)
    _p0_Q=np.random.randint(1,100,nwalkers)
    p0=[d for d in zip(_p0_D,_p0_Q)]
    #pars = self.ini_mod  # initial guess
    #p0 = np.array(np.abs([pars + 10.*np.random.randn(ndim) for i in range(nwalkers)]))  # add some noise
    #p0 = np.abs(p0)
    # set the emcee sampler to start at the initial guess and run 5000 burn-in jumps
    #sq_prof_est =  self.square_fn(pars, x_axis=self.meb_depth_rs, y_base = 2.)
    pos,prob,state=sampler.run_mcmc(p0,walk_jump)
    f = open("chain.dat", "w")
    nk,nit,ndim=sampler.chain.shape
    for k in range(nk):
        for i in range(nit):
            f.write("{:d} {:d} ".format(k, i))
            for j in range(ndim):
                f.write("{:15.7f} ".format(sampler.chain[k,i,j]))
            f.write("{:15.7f}\n".format(sampler.lnprobability[k,i]))
    f.close()
def plot_results_mcmc(fl=None,corner_plt=True, walker_plt=True, par_dist=True): 
    if fl is None:
        chain = np.genfromtxt('chain.dat')
    else:
        chain = np.genfromtxt(fl)
    if corner_plt: 
    # show corner plot
        weights = chain[:,-1]
        weights -= np.max(weights)
        weights = np.exp(weights)
        labels = ['D','Q']
        fig = corner.corner(chain[:,2:-1], labels=labels, weights=weights, smooth=1, bins=30)
        plt.savefig('corner_plot.png', dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)
        #plt.close(fig)
    if walker_plt:
        labels = ['D','Q']
        npar = int(chain.shape[1] - 3)
        f,axs = plt.subplots(npar,1)
        f.set_size_inches([8,8])
        for i,ax,label in zip(range(npar),axs,labels):
            for j in np.unique(chain[:,0]):
                ind = np.where(chain[:,0] == j)
                it = chain[ind,1]
                par = chain[ind,2+i]
                ax.plot(it[0],par[0],'k-')
            ax.set_ylabel(label)
        plt.savefig('walkers.png', dpi=300)
    if par_dist:
        labels = ['D','Q']
        npar = int(chain.shape[1] - 3)
        f,axs = plt.subplots(1,npar)
        f.set_size_inches([8,4])
        for i,ax,label in zip(range(npar),axs,labels):
            _L = chain[:,i+2]
            bins = np.linspace(np.min(_L), np.max(_L), int(np.sqrt(len(_L))))
            h,e = np.histogram(_L, bins, density = True)
            m = 0.5*(e[:-1]+e[1:])
            ax.bar(e[:-1], h, e[1]-e[0])#, label = 'histogram')
            ax.set_xlabel(labels[i])
            ax.set_ylabel('freq.')
            ax.grid(True, which='both', linewidth=0.1)
        plt.savefig('par_dist.png', dpi=300)
    chain = None
    plt.tight_layout()
    plt.show()
    #plt.close('all')
def read_chain_pars():
    '''
    Return percetiles 10,50,90 for each parameters
    i.e. for 2 pars
    [[p10,p50,p90],[p10,p50,p90]]
    '''
    chain = np.genfromtxt('chain.dat')
    npar = int(chain.shape[1] - 3)
    pars = [None] * npar
    for i in range(npar):
        _L = chain[:,i+2]
        #pars[i]=[np.percentile(_L,10),np.percentile(_L,50),np.percentile(_L,90)]
        pars[i]=[np.mean(_L),np.std(_L)]
    return pars
#
def plot_pars(fit=True):
    df = pd.read_csv('df_pars.csv', index_col= 'time', infer_datetime_format=True)
    # Converting the index as date
    df.index = pd.to_datetime(df.index)
    #
    labels = ['D','Q']
    npar = int((df.shape[1]-1)/2)
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(12, 7))
    #rsam
    ax0.plot(df.index, df['rsamF'],label='RSAM',color='b')
    ax1.errorbar(x=df.index,y=df['Qm'],yerr=df['Qs']/2,label='Q: mean gas flux',color='r', ecolor='r')
    ax2.errorbar(x=df.index,y=df['Dm'],yerr=df['Ds']/2,label='D: gas pocket thickness',color='g', ecolor='g')
    if fit:
        f_l=[]
        for i in range(len(df)):
            DP, DP2, U_z, V_z = _4mcmc_fwd_cav_Ps_Uz_Vz(D=df['Dm'][i], Q=df['Qm'][i])
            f_l.append(np.mean(np.abs(V_z)*1e6))
        mf=np.sum(sqrt((f_l-df['rsamF'])**2))/len(df)
        ax0.plot(df.index, f_l,'k--',label='est_RSAM (L2: '+str(round(mf,1))+')')
    ax0.legend()
    ax1.legend()
    ax2.legend()
    ax0.set_ylabel('RSAM')
    ax1.set_ylabel('Q [kg/s]')
    ax2.set_ylabel('D [m]')

    ax0.set_yscale('log')

    plt.savefig('pars_fit.png', dpi=300)
def re_read_chains(fl_df):
    '''
    re read chains to compute percentiles and add to df pars
    '''
    df = pd.read_csv(fl_df, index_col= 'time', infer_datetime_format=True)
    # Converting the index as date
    df.index = pd.to_datetime(df.index)
    #
    p10_l, p20_l, p30_l, p40_l, p50_l, p60_l, p70_l, p80_l, p90_l = [],[],[],[],[],[],[],[],[]
    for i in range(len(df)):
        # read chain
        chain = np.genfromtxt('./chains/chain'+str(i)+'.dat')
        npar = int(chain.shape[1] - 3)
        pars = [None] * npar
        for j in [3]:#range(npar): # just Q
            _L = chain[:,j]#[:,j+2]
            p10_l.append(np.percentile(_L,10))
            p20_l.append(np.percentile(_L,20))
            p30_l.append(np.percentile(_L,30))
            p40_l.append(np.percentile(_L,40))
            p50_l.append(np.percentile(_L,50))
            p60_l.append(np.percentile(_L,60))
            p70_l.append(np.percentile(_L,70))
            p80_l.append(np.percentile(_L,80))
            p90_l.append(np.percentile(_L,90))
    df['Q_p10'] = [round(i,3) for i in p10_l]
    df['Q_p20'] = [round(i,3) for i in p20_l]
    df['Q_p30'] = [round(i,3) for i in p30_l]
    df['Q_p40'] = [round(i,3) for i in p40_l]
    df['Q_p50'] = [round(i,3) for i in p50_l]
    df['Q_p60'] = [round(i,3) for i in p60_l]
    df['Q_p70'] = [round(i,3) for i in p70_l]
    df['Q_p80'] = [round(i,3) for i in p80_l]
    df['Q_p90'] = [round(i,3) for i in p90_l]
    #
    df.to_csv('df_pars_pers.csv')
def plot_pars_pers(fit=True):
    df = pd.read_csv('df_pars_pers.csv', index_col= 'time', infer_datetime_format=True)
    # Converting the index as date
    df.index = pd.to_datetime(df.index)
    #
    npar = int((df.shape[1]-1)/2)
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(12, 7))
    #rsam
    ax0.plot(df.index, df['rsamF'],label='RSAM',color='b')
    ax1.errorbar(x=df.index,y=df['Qm'],yerr=df['Qs']/2,label='Q: mean gas flux',color='r', ecolor='r')
    ax2.errorbar(x=df.index,y=df['Dm'],yerr=df['Ds']/2,label='D: gas pocket thickness',color='g', ecolor='g')
    #
    ax1.plot(df.index, df['Q_p10'],label='p10',color='m')
    ax1.plot(df.index, df['Q_p50'],label='p50',color='b')
    ax1.plot(df.index, df['Q_p90'],label='p90',color='g')
    #
    if fit:
        f_l=[]
        for i in range(len(df)):
            DP, DP2, U_z, V_z = _4mcmc_fwd_cav_Ps_Uz_Vz(D=df['Dm'][i], Q=df['Qm'][i])
            f_l.append(np.mean(np.abs(V_z)*1e6))
        mf=np.sum(sqrt((f_l-df['rsamF'])**2))/len(df)
        ax0.plot(df.index, f_l,'k--',label='est_RSAM (L2: '+str(round(mf,1))+')')
    ax0.legend()
    ax1.legend()
    ax2.legend()
    ax0.set_ylabel('RSAM')
    ax1.set_ylabel('Q [kg/s]')
    ax2.set_ylabel('D [m]')

    #ax0.set_yscale('log')

    plt.savefig('pars_fit_pers.png', dpi=300)
    
##########################################################
def main():
    ##
    if True : # run mcmc
        import time as _time
        st = _time.time()
        ##
        #ex1 (1.5 days, 1 s/hour, 8 hours run)
        #ti='2016-11-13 12:00:00'
        #tf='2016-11-15 00:00:00'
        #ex2 (2 days, .5 s/hour, 8 hours run)
        #ti='2016-11-12 12:00:00'
        #tf='2016-11-14 12:00:00'
        #ex3 (.5 days, 1 s/hour,  hours run)
        ti='2016-11-13 12:00:00'
        tf='2016-11-14 00:00:00'
        df = _load_rsam_data(ti,tf, downsample=6)
        #
        Qm_l,Qs_l,Dm_l,Ds_l=[],[],[],[]
        for i in range(len(df)):
            run_mcmc(df['rsamF'][i], walk_jump = 200)
            pars=read_chain_pars()
            #
            Qm_l.append(pars[1][0])
            Qs_l.append(pars[1][1])
            Dm_l.append(pars[0][0])
            Ds_l.append(pars[0][1])
            os.replace("chain.dat", "./chains/chain"+str(i)+".dat")
            #
        df['Dm'] = [round(i,3) for i in Dm_l]
        df['Ds'] = [round(i,3) for i in Ds_l]
        df['Qm'] = [int(i) for i in Qm_l]
        df['Qs'] = [int(i) for i in Qs_l]
        #
        df.to_csv('df_pars.csv')
        ###########################
        # get the end time
        et = _time.time()
        # get the execution time
        elapsed_time = et - st
        print('Execution time:', elapsed_time, 'seconds')
        ##############################
    if True:
        pass
        plot_pars()
        plot_results_mcmc(fl='./chains/chain2.dat') 
        re_read_chains(fl_df='df_pars.csv')
        plot_pars_pers()

if __name__ == "__main__":
    main()