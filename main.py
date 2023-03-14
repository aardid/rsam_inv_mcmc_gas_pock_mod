#############################################################################
import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd 
import math, random
import matplotlib.pyplot as plt
import emcee, corner
import multiprocessing 
from multiprocessing import Pool
from datetime import datetime, timedelta, date
from pandas._libs.tslibs.timestamps import Timestamp
from obspy import UTCDateTime, read_inventory
from scipy.interpolate import RegularGridInterpolator
from obspy.signal.filter import bandpass 
from scipy.integrate import cumtrapz
import seaborn as sns
import time
import os
# define constant
pi=np.pi
sqrt=np.sqrt
_MONTH = timedelta(days=365.25/12)
_DAY = timedelta(days=1.)
_HOUR = timedelta(days=1./24)
_MIN = timedelta(days=1./24/60)
# global variables 
BANDS=['rsam','mf','hf']#=['vlf','lf','rsam','mf','hf']
RATIOS=None#['dsar']
FILTERED_DATA=True # EQ filtered data. Add 'F' when loading
FBANDS=[[2, 5], [4.5, 8], [8,16]] #FBANDS=[[0.01,0.1],[0.1,2],[2, 5], [4.5, 8], [8,16]]
lookup_table=True
# parameteres priors (pars are D, Q, N)
pars_priors = [[0.001,10],[1,100],[10,10000]]
if lookup_table:
        _dir = 'lookup_tables'
        eos_rsam=np.load(_dir+os.sep+'eos_rsam.npy')
        eos_mf=np.load(_dir+os.sep+'eos_mf.npy')
        eos_hf=np.load(_dir+os.sep+'eos_hf.npy')
        eos_dp=np.load(_dir+os.sep+'eos_dp.npy')
        if RATIOS:
            eos_rmar=np.load(_dir+os.sep+'eos_rmar.npy')
            eos_dsar=np.load(_dir+os.sep+'eos_dsar.npy')

        # list of parameter dimensions
        D=3   # number of dimensions
        nD=np.shape(eos_rsam)[0]#10 # number of parameter divisions in each dimension
        _D=np.logspace(-3, 1, num=nD)
        _Q=np.linspace(1,100,nD)
        _N=np.logspace(0, 4, num=nD)
        #  create look-up table
        ps = [_D,_Q,_N]
        rgi_rsam=RegularGridInterpolator(ps, eos_rsam)    
        rgi_mf=RegularGridInterpolator(ps, eos_mf)
        rgi_hf=RegularGridInterpolator(ps, eos_hf)
        rgi_dp=RegularGridInterpolator(ps, eos_dp) 
        if RATIOS:
            rgi_rmar=RegularGridInterpolator(ps, eos_rmar)
            rgi_dsar=RegularGridInterpolator(ps, eos_dsar)   
##############################################################################
def datetimeify(t):
    """ Return datetime object corresponding to input string.
        Parameters:
        -----------
        t : str, datetime.datetime
            Date string to convert to datetime object.
        Returns:
        --------
        datetime : datetime.datetime
            Datetime object corresponding to input string.
        Notes:
        ------
        This function tries several datetime string formats, and raises a ValueError if none work.
    """
    if type(t) in [datetime, Timestamp]:
        return t
    if type(t) is UTCDateTime:
        return t._get_datetime()
    fmts = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y %m %d %H %M %S','%Y%m%d:%H%M',]
    for fmt in fmts:
        try:
            return datetime.strptime(t, fmt)
        except ValueError:
            pass
    raise ValueError("time data '{:s}' not a recognized format".format(t))
def _period_into_months(begin,end,verbose=False):
    '''
    devide period into a list of monthly periods.
    begin = '%Y-%m-%d'
    end = '%Y-%m-%d'

    Example:
    Input
        begin = '2011-01-01'
        end = '2011-02-12'
    Outout:
        periods = [[datetime('2011-01-01 00:00'), datetime('2011-01-31 23:50'')]
            ,[[datetime('2011-02-01 00:00'), datetime('2011-02-12 23:50')]]

    '''
    dt_start = datetime.strptime(begin, '%Y-%m-%d')
    dt_end = datetime.strptime(end, '%Y-%m-%d')
    one_day = timedelta(1)
    start_dates = [dt_start]
    end_dates = []
    today = dt_start
    while today <= dt_end:
        #print(today)
        tomorrow = today + one_day
        if tomorrow.month != today.month:
            start_dates.append(tomorrow)
            end_dates.append(today)
        today = tomorrow
    end_dates.append(dt_end)
    out_fmt = '%d %B %Y'
    periods = []
    for start, end in zip(start_dates,end_dates):
        periods.append([start, end+23*_HOUR+50*_MIN])
        if verbose:
            print('{} to {}'.format(start.strftime(out_fmt), end.strftime(out_fmt)))   
    return periods
##############################################################################
def _load_tremor_data(ti,tf, downsample=None, fl=None):
    '''
    Returns a data frame of RSAM between dates ti and tf (default from Ruapehu data)
    i.e.,
    ti='2016-11-13 12:00:00'
    tf='2016-11-28'
    downsample: value given correspond to the step size. For example, if the original sampling is
    every 10 mins, a step size of 6 will resample to every hour

    Note: tremor in amplitude measurementes (e.g., rsam) is divided by 1e9 for taking it to m/s units (SI)
    '''
    # import RSAM Ruapehu data
    usecols=[]
    usecols.append('time')
    if FILTERED_DATA:
        [usecols.append(b+'F') for b in BANDS]
        if RATIOS:
            [usecols.append(b+'F') for b in RATIOS]
    else:
        [usecols.append(b) for b in BANDS]
        if RATIOS:
            [usecols.append(b) for b in RATIOS]
    if fl is None:
        fl= 'C:'+os.sep+'Users'+os.sep+'aar135'+os.sep+'codes_local_disk'+os.sep+'volc_forecast_tl'+os.sep+'volc_forecast_tl'+os.sep+'data'+os.sep+'FWVZ_tremor_data.csv'
    df = pd.read_csv(fl, index_col= 'time', usecols=usecols, infer_datetime_format=True)
    # Filter between dates 
    for column in df:
        if column in ['vlf','lf','rsam','mf','hf','vlfF','lfF','rsamF','mfF','hfF']:
            df[column] = df[column].apply(lambda x: x*1e-9)
    try:
        df = df.loc[ti:tf]
        df.index = pd.to_datetime(df.index)
    except:
        df.index = pd.to_datetime(df.index)
        df = df.loc[ti:tf]
    if downsample:
        step_size=downsample
        df = df.iloc[::step_size, :]
    return df
def fwd_cav_Ps_Uz_Vz(D=None,Q=None,N=None, Pressure=None):
    '''
    Forward function from Girona et al(2019) to compute DP, DP2, U_z, V_z
    Q   # mean gas flux (e.g. 500)
    D   # gas pocket thickness (e.g. 0.5)
    N   # number of mass impulses in tau seconds  (e.g. 1000)
    Pressure: compute pressure as output

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
    #N=1000;#1                                                       #number of mass impulses in tau seconds

    # FWVZ
    if False:
        tau=30;#600                                                     #seconds of simulation
        R=20;                                                          #conduit radius
        L=100;                                                           #thickness of the cap
        distance=2000;                                                  #source-receiver distance
        max_freq=50;                                                    #maximum frequency to be reached in the simulation
    # WIZ
    if True:
        tau=30;#600                                                     #seconds of simulation
        R=10;                                                          #conduit radius
        L=100;                                                           #thickness of the cap
        distance=1000;                                                  #source-receiver distance
        max_freq=50;                                                    #maximum frequency to be reached in the simulation

    ## PHYSICAL PARAMETERS
    S=pi*R**2;                                                                  #conduit section
    mu_g=1e-5;                                                                  #gas viscosity
    T=1000+273.15;                                                              #gas temperature
    M=0.018;                                                                    #molecular weight of gas (water vapor)
    Rg=8.3145;                                                                  #ideal gas constant
    Pex=101325;                                                                 #external pressure
    kappa=1e-8;#1e-8;                                                                 #permeability of the cap
    phi=0.0001;                                                                 #porosity of the cap
    rho_s=3000;                                                                 #density of the medium of propagation
    Qf=50;#40#20;                                                                      #quality factor. 
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
    if Pressure:
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
def Vz_filters(Vz, sampfreq = 50):
    '''
    Filter Vz into frequency bands. 
    Bands are dfine as global variable (see top) between:
        BANDS=['vlf','lf','rsam','mf','hf']
        FBANDS=[[0.01,0.1],[0.1,2],[2, 5], [4.5, 8], [8,16]]
    Input:
        Vz: velocity time serie
    Returns:
        _datas : [[]...[]] list of arrays of filtered timeseries  
    '''
    # filter and compute RSAM, MF, anf HF
    #BANDS=['vlf','lf','rsam','mf','hf']
    #FBANDS=[[0.01,0.1],[0.1,2],[2, 5], [4.5, 8], [8,16]]
    _datas = []
    for fmin,fmax in FBANDS:
        _data=abs(bandpass(Vz, fmin, fmax, sampfreq))
        _datas.append(_data)#*1e6)
    if False:
        f, axes = plt.subplots(len(_datas), 1)
        for i in range(len(_datas)):
            axes[i].plot(_datas[i], label=BANDS[i])
            axes[i].legend(loc=1)
            plt.show()
    #
    return _datas
def calc_ratios(U_z, sampfreq = 50):
    '''
    Filter Uz in frequency bands and compute ratios RMAR (rsam/mf) and DSAR (mf/hf)
    '''
    _datas = []
    _datas_r = []
    for fmin,fmax in FBANDS:
        _data=abs(bandpass(U_z, fmin, fmax, sampfreq))
        _datas.append(_data)#*1e6)
    #
    _datas_r.append(_datas[BANDS.index("rsam")]/_datas[BANDS.index("mf")])
    _datas_r.append(_datas[BANDS.index("mf")]/_datas[BANDS.index("hf")])
    return _datas_r

## MCMC functions
def prob_likelihood(est, obs):
    v=1#np.std(obs)#1
    norm=2
    est=np.array(est)
    obs=np.array(obs)
    if len(obs) > 1:
        if RATIOS:
            est[0:3]=np.log10(est[0:3]*1e9)
            obs[0:3]=np.log10(obs[0:3]*1e9)
            est[-2:]=np.log10(est[-2:])
            obs[-2:]=np.log10(obs[-2:])
            weigths = np.array([5,1,1,0,0])
            prob= -.5* np.sum(weigths*(((obs-est)/(0.05*obs))**norm))
        else:
            est=np.log10(np.array(est)*1e9)
            obs=np.log10(np.array(obs)*1e9)
            weigths = np.array([1.,.1,.1])
            prob= -.5* np.sum(weigths*(((obs-est)/(0.01*obs))**norm))
        #prob= -.5* np.sum(weigths*((obs-est)**norm))
    else:
        prob = -.5*(np.sum((est - obs)**norm))/(v**norm)
    return prob
def lnprob(pars, obs):
    ## Parameter constraints
    if (any(x<=0 for x in pars)): # positive parameters
        return -np.Inf
    # D
    if pars[0] < pars_priors[0][0]:#0.001:# max(self.meb_prof)+5.: # percentage range 
        return -np.Inf
    if pars[0] > pars_priors[0][1]:#1.:# max(self.meb_prof)+5.: # percentage range 
        return -np.Inf
    # Q
    if pars[1] < pars_priors[1][0]:#1:# max(self.meb_prof)+5.: # percentage range 
        return -np.Inf
    if pars[1] > pars_priors[1][1]:#100:# max(self.meb_prof)+5.: # percentage range 
        return -np.Inf
    # N
    if pars[2] < pars_priors[2][0]:#1e1:#4990:#1e1:# max(self.meb_prof)+5.: # percentage range 
        return -np.Inf
    if pars[2] > pars_priors[2][1]:#1e4:#5010:#1e4:# max(self.meb_prof)+5.: # percentage range 
        return -np.Inf
    if int(pars[2]) == 0:# max(self.meb_prof)+5.: # percentage range 
        return -np.Inf
    #
    if lookup_table:
        p=np.array([pars[0],pars[1],pars[2]])    # random parameter vector
        if RATIOS:
            est = [rgi_rsam(p)[0],rgi_mf(p)[0],rgi_hf(p)[0],rgi_rmar(p)[0],rgi_dsar(p)[0]] # look up simulation result
        else:   
            est = [rgi_rsam(p)[0],rgi_mf(p)[0],rgi_hf(p)[0]] # look up simulation result
    else:
        ## estimate square function of clay content given pars [z1,z2,%] 
        DP, DP2, U_z, V_z = fwd_cav_Ps_Uz_Vz(pars[0],int(pars[1]),int(pars[2]))
        #est = np.mean(np.abs(V_z)*1e6)
        _datas= Vz_filters(V_z)
        if RATIOS:
            _datas_r= calc_ratios(U_z)
            _datas=_datas+_datas_r
        est = [np.median(np.abs(_d)) for _d in _datas]    
    ## calculate prob and return  
    prob = prob_likelihood(est,obs) # micro m/s (*1e6)
    ## check if prob is nan
    if prob!=prob: # assign non values to -inf
        return -np.Inf
    return prob
def lnprob_bkp(pars, obs):
    ## Parameter constraints
    if (any(x<=0 for x in pars)): # positive parameters
        return -np.Inf
    # D
    if pars[0] > 5.:# max(self.meb_prof)+5.: # percentage range 
        return -np.Inf
    # Q
    if pars[1] < 1:# max(self.meb_prof)+5.: # percentage range 
        return -np.Inf
    if pars[1] > 100:# max(self.meb_prof)+5.: # percentage range 
        return -np.Inf
    # N
    if pars[2] < 1e1:# max(self.meb_prof)+5.: # percentage range 
        return -np.Inf
    if pars[2] > 1e4:# max(self.meb_prof)+5.: # percentage range 
        return -np.Inf
    if int(pars[2]) == 0:# max(self.meb_prof)+5.: # percentage range 
        return -np.Inf
    #

    ## estimate square function of clay content given pars [z1,z2,%] 
    DP, DP2, U_z, V_z = fwd_cav_Ps_Uz_Vz(pars[0],int(pars[1]),int(pars[2]))
    #est = np.mean(np.abs(V_z)*1e6)
    _datas= Vz_filters(V_z)
    est = [np.mean(np.abs(_d)) for _d in _datas]    

    ## calculate prob and return  
    prob = prob_likelihood(est,obs) # micro m/s (*1e6)
    ## check if prob is nan
    if prob!=prob: # assign non values to -inf
        return -np.Inf
    return prob 
def run_mcmc(data_p, walk_jump = None, _dir=None):
    '''
    run mcmc. Results are save in local directory in chain.dat file
    data_p: list of values to to fit 
    _dir: directory to save files (None for current directory)
    '''
    nwalkers= 20        # number of walkers
    ndim = 3               # parameter space dimensionality
    if walk_jump is None:
        walk_jump = 200
    ## Timing inversion
    #start_time = time.time()
    # create the emcee object (set threads>1 for multiprocessing)
    data = data_p#df['rsam'][0]# U_z_noise
    cores = multiprocessing.cpu_count()
    # Create sampler
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[data], threads=4, pool=pool)
        # set the initial location of the walkers
        _p0_D=np.random.uniform(pars_priors[0][0],pars_priors[0][1],nwalkers)
        _p0_Q=np.random.randint(pars_priors[1][0],pars_priors[1][1],nwalkers)
        _p0_N=np.random.randint(pars_priors[2][0],pars_priors[2][1],nwalkers)#4990,5010,nwalkers)#1e1,1e4,nwalkers)
        p0=[d for d in zip(_p0_D,_p0_Q,_p0_N)]
        #pars = self.ini_mod  # initial guess
        #p0 = np.array(np.abs([pars + 10.*np.random.randn(ndim) for i in range(nwalkers)]))  # add some noise
        #p0 = np.abs(p0)
        # set the emcee sampler to start at the initial guess and run 5000 burn-in jumps
        #sq_prof_est =  self.square_fn(pars, x_axis=self.meb_depth_rs, y_base = 2.)
        pos,prob,state=sampler.run_mcmc(p0,walk_jump) #progress=True)
        if _dir:
            f = open(_dir+os.sep+"chain.dat", "w")
        else:
            f = open("chain.dat", "w")
        nk,nit,ndim=sampler.chain.shape
        for k in range(nk):
            for i in range(nit):
                f.write("{:d} {:d} ".format(k, i))
                for j in range(ndim):
                    f.write("{:15.7f} ".format(sampler.chain[k,i,j]))
                f.write("{:15.7f}\n".format(sampler.lnprobability[k,i]))
        f.close()
def run_mcmc_aux(df, _dir = None):
    '''
    Function that run mcmc for several points given in df (dataframe) 
    and creates a df of results.
    
    _dir: directory to save files (None for current directory)    
    '''
    # check directory structure
    newpath = r'.'+os.sep+'chains' 
    if _dir: 
        newpath = r'.'+os.sep+_dir+os.sep+'chains' 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    Qm_l,Qs_l,Dm_l,Ds_l,Nm_l,Ns_l=[],[],[],[],[],[]
    for i in range(len(df)):
        #run mcmc
        #run_mcmc(df['rsamF'][i], walk_jump = 300, _dir=_dir)
        if FILTERED_DATA:
            if RATIOS:
                _points = [df[_b+'F'][i] for _b in BANDS+RATIOS] 
            else:
                _points = [df[_b+'F'][i] for _b in BANDS] 
        else:
            if RATIOS:
                _points = [df[_b][i] for _b in BANDS+RATIOS] 
            else:
                _points = [df[_b][i] for _b in BANDS] 
        run_mcmc(_points, walk_jump = 500, _dir=_dir)
        #
        pars=read_chain_pars(_dir=_dir)
        #
        Qm_l.append(pars[1][0])
        Qs_l.append(pars[1][1])
        Dm_l.append(pars[0][0])
        Ds_l.append(pars[0][1])
        Nm_l.append(pars[2][0])
        Ns_l.append(pars[2][1])
        if _dir:
            os.replace(_dir+os.sep+"chain.dat", _dir+os.sep+"chains"+os.sep+"chain"+str(i)+".dat")
        else:
            os.replace("chain.dat", "./chains/chain"+str(i)+".dat")
        #print(str(i+1)+'/'+str(len(df)))
        #
    df['Dm'] = [round(i,3) for i in Dm_l]
    df['Ds'] = [round(i,3) for i in Ds_l]
    df['Qm'] = [int(i) for i in Qm_l]
    df['Qs'] = [int(i) for i in Qs_l]
    df['Nm'] = [int(i) for i in Nm_l]
    df['Ns'] = [int(i) for i in Ns_l]
    #
    if _dir:
        df.to_csv(_dir+os.sep+'df_pars.csv')    
    else:
        df.to_csv('df_pars.csv')
    # compute percetiles 
    if True: 
        if _dir:
            re_read_chains(_dir=_dir)
        else:
            re_read_chains()      
    #
# Plotting functions 
def plot_results_mcmc(fl=None,corner_plt=True, walker_plt=True, par_dist=True): 
    if fl is None:
        chain = np.genfromtxt('chain.dat')
    else:
        chain = np.genfromtxt(fl)
    if False: # filter burn-out section
        _ =[]
        for i in range(len(chain)-1):
            if chain[i][1] < 50:
                _.append(i)
        #for i in _:
        chain = np.delete(chain, _, axis=0)
        #
    if corner_plt: 
    # show corner plot
        weights = chain[:,-1]
        weights -= np.max(weights)
        weights = np.exp(weights)
        labels = ['D','Q','N']
        fig = corner.corner(chain[:,2:-1], labels=labels, weights=weights, smooth=1, bins=30)
        plt.savefig('corner_plot.png', dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', format='png',transparent=True, bbox_inches=None, pad_inches=0.1)
        #plt.close(fig)
    if walker_plt:
        labels = ['D','Q','N']
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
        labels = ['D','Q','N']
        npar = int(chain.shape[1] - 3)
        f,axs = plt.subplots(1,npar)
        f.set_size_inches([8,4])
        for i,ax,label in zip(range(npar),axs,labels):
            _L = chain[:,i+2]
            bins = np.linspace(np.min(_L), np.max(_L), int(np.sqrt(len(_L))))
            h,e = np.histogram(_L, bins, density = True)
            #m = 0.5*(e[:-1]+e[1:])
            ax.bar(e[:-1], h, e[1]-e[0])#, label = 'histogram')
            ax.set_xlabel(labels[i])
            ax.set_ylabel('freq.')
            ax.grid(True, which='both', linewidth=0.1)
        plt.savefig('par_dist.png', dpi=300)
    chain = None
    plt.tight_layout()
    plt.show()
    #plt.close('all')
def read_chain_pars(_dir=None):
    '''
    Return percetiles 10,50,90 for each parameters
    i.e. for 2 pars
    [[p10,p50,p90],[p10,p50,p90]]

    _dir: directory to load files (None for current directory)
    '''
    if _dir:
        chain = np.genfromtxt(_dir+os.sep+'chain.dat')
    else:
        chain = np.genfromtxt('chain.dat')
    npar = int(chain.shape[1] - 3)
    pars = [None] * npar
    for i in range(npar):
        _L = chain[:,i+2]
        #pars[i]=[np.percentile(_L,10),np.percentile(_L,50),np.percentile(_L,90)]
        pars[i]=[np.mean(_L),np.std(_L)]
    return pars
def plot_pars(_dir=None, fit=True, Pressure = None, plot_erup=None):
    '''
    _dir: directory to save files (None for current directory)
    '''
    if _dir:
        df = pd.read_csv(_dir+os.sep+'df_pars.csv', index_col= 'time', infer_datetime_format=True)
    else:
        df = pd.read_csv('df_pars.csv', index_col= 'time', infer_datetime_format=True)
    # Converting the index as date
    df.index = pd.to_datetime(df.index)
    ## plot in mi m/s (1e6 ms) (mu)
    mi=1.e6 # 1.
    #
    labels = ['D','Q','N']
    npar = int((df.shape[1]-1)/2)
    if RATIOS:
        if len(BANDS+RATIOS) == 1:
            fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(12, 10))
            #rsam
            ax0.plot(df.index, df['rsamF']*mi,label='RSAM',color='b')
            ax1.errorbar(x=df.index,y=df['Qm'],yerr=df['Qs']/2,label='Q: mean gas flux',color='r', ecolor='r')
            ax2.errorbar(x=df.index,y=df['Dm'],yerr=df['Ds']/2,label='D: gas pocket thickness',color='g', ecolor='g')
            ax3.errorbar(x=df.index,y=df['Nm'],yerr=df['Ns']/2,label='N: Nomber of impulses',color='m', ecolor='m')
            if fit:
                f_l=[]
                for i in range(len(df)):
                    DP, DP2, U_z, V_z = fwd_cav_Ps_Uz_Vz(D=df['Dm'][i], Q=df['Qm'][i], N=df['Nm'][i])
                    f_l.append(np.mean(np.abs(V_z)))
                mf=np.sum(sqrt((f_l*mi-df['rsamF']*mi)**2))/len(df)
                ax0.plot(df.index, f_l*mi,'k--',label='est_RSAM (L2: '+str(round(mf,1))+')')
            ax0.legend(loc=1)
            ax1.legend(loc=1)
            ax2.legend(loc=1)
            ax0.set_ylabel('RSAM')
            ax1.set_ylabel('Q [kg/s]')
            ax2.set_ylabel('D [m]')
            ax3.set_ylabel('N')
        if len(BANDS+RATIOS) == 3:
            if Pressure:
                fig, (ax0, ax2, ax4, ax1, ax3, ax5, ax6) = plt.subplots(7, 1, figsize=(18, 12))
            else:
                fig, (ax0, ax2, ax4, ax1, ax3, ax5) = plt.subplots(6, 1, figsize=(16, 12))
            #rsam
            if FILTERED_DATA:
                ax0.plot(df.index, df[BANDS[0]+'F']*mi,label=BANDS[0]+'F',color='b')
                ax2.plot(df.index, df[BANDS[1]+'F']*mi,label=BANDS[1]+'F',color='b')
                ax4.plot(df.index, df[BANDS[2]+'F']*mi,label=BANDS[2]+'F',color='b')

                ax1.errorbar(x=df.index,y=df['Qm'],yerr=df['Qs']/2,label='Q: mean gas flux',color='r', ecolor='r')
                ax3.errorbar(x=df.index,y=df['Dm'],yerr=df['Ds']/2,label='D: gas pocket thickness',color='g', ecolor='g')
                ax5.errorbar(x=df.index,y=df['Nm'],yerr=df['Ns']/2,label='N: Nomber of impulses',color='m', ecolor='m')
            #
            if fit:
                if len(BANDS)==1:
                    f_l=[]
                    for i in range(len(df)):
                        DP, DP2, U_z, V_z = fwd_cav_Ps_Uz_Vz(D=df['Dm'][i], Q=df['Qm'][i], N=df['Nm'][i])
                        f_l.append(np.mean(np.abs(V_z)))
                    mf=np.sum(sqrt((f_l*mi-df['rsamF']*mi)**2))/len(df)
                    ax0.plot(df.index, f_l*mi,'k--',label='est_RSAM (L2: '+str(round(mf,1))+')')
                    #
                    ax0.legend(loc=1)
                    ax1.legend(loc=1)
                    ax2.legend(loc=1)
                    ax0.set_ylabel('RSAM')
                    ax1.set_ylabel('Q [kg/s]')
                    ax2.set_ylabel('D [m]')
                    ax3.set_ylabel('N')
                if len(BANDS)==3:
                    f_l_0,f_l_1,f_l_2,f_l_p=[],[],[],[]
                    for i in range(len(df)):
                        #DP, DP2, U_z, V_z = fwd_cav_Ps_Uz_Vz(D=df['Dm'][i], Q=df['Qm'][i], N=df['Nm'][i])
                        #_datas= Vz_filters(V_z)
                        #
                        if (lookup_table):# and not Pressure):
                            #
                            D=df['Dm'][i]
                            Q=df['Qm'][i]
                            N=df['Nm'][i]
                            p=np.array([D,Q,N])    # random parameter vector
                            est = [rgi_rsam(p)[0],rgi_mf(p)[0],rgi_hf(p)[0]] # look up simulation result
                            f_l_0.append(est[0])
                            f_l_1.append(est[1])
                            f_l_2.append(est[2])
                            if Pressure:
                                f_l_p.append(rgi_dp(p)[0])
                        else:
                            ## estimate square function of clay content given pars [z1,z2,%] 
                            if Pressure:
                                DP, DP2, U_z, V_z = fwd_cav_Ps_Uz_Vz(D=df['Dm'][i], Q=df['Qm'][i], N=df['Nm'][i], Pressure=True)
                            else:
                                DP, DP2, U_z, V_z = fwd_cav_Ps_Uz_Vz(D=df['Dm'][i], Q=df['Qm'][i], N=df['Nm'][i])
                            #est = np.mean(np.abs(V_z)*1e6)
                            _datas= Vz_filters(V_z)
                            #est = [np.mean(np.abs(_d)) for _d in _datas] 
                            #for _d in _datas:
                            f_l_0.append(np.mean(np.abs(_datas[0])))
                            f_l_1.append(np.mean(np.abs(_datas[1])))
                            f_l_2.append(np.mean(np.abs(_datas[2])))
                            if Pressure:
                                f_l_p.append(np.mean(np.real(DP)))
                            #
                    f_l_0=np.asarray(f_l_0)
                    f_l_1=np.asarray(f_l_1)
                    f_l_2=np.asarray(f_l_2)
                    #
                    mf=np.sum(sqrt((f_l_0*mi-df['rsamF']*mi)**2))/len(df)
                    ax0.plot(df.index, f_l_0*mi,'k--',label='est_RSAM (L2: '+str(round(mf,1))+')')

                    mf=np.sum(sqrt((f_l_1*mi-df['hfF']*mi)**2))/len(df)
                    ax2.plot(df.index, f_l_1*mi,'k--',label='est_MF (L2: '+str(round(mf,1))+')')

                    mf=np.sum(sqrt((f_l_2*mi-df['hfF']*mi)**2))/len(df)
                    ax4.plot(df.index, f_l_2*mi,'k--',label='est_HF (L2: '+str(round(mf,1))+')')

                    if Pressure:
                        #mf=np.sum(sqrt((f_l_2-df['hfF'])**2))/len(df)
                        ax6.plot(df.index, np.array(f_l_p),'k--',label='Pressure')
                    #
                    ax0.legend(loc=1)
                    ax1.legend(loc=1)
                    ax2.legend(loc=1)
                    ax3.legend(loc=1)
                    ax4.legend(loc=1)
                    ax5.legend(loc=1)
                    if mi != 1.:
                        ax0.set_ylabel('RSAM [1e6 m/s]')
                        ax2.set_ylabel('MF [1e6 m/s]')
                        ax4.set_ylabel('HF [1e6 m/s]')
                    ax1.set_ylabel('Q [kg/s]')
                    ax3.set_ylabel('D [m]')
                    ax5.set_ylabel('N')
                    if Pressure:
                        ax6.set_ylabel('P [Pa]')
        if len(BANDS+RATIOS) == 5:
            if Pressure:
                fig, (ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(9, 1, figsize=(20, 12))
            else:
                fig, (ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(8, 1, figsize=(18, 12))
            #rsam
            if FILTERED_DATA:
                ax0.plot(df.index, df[BANDS[0]+'F']*mi,label=BANDS[0]+'F',color='b')
                ax1.plot(df.index, df[BANDS[1]+'F']*mi,label=BANDS[1]+'F',color='b')
                ax2.plot(df.index, df[BANDS[2]+'F']*mi,label=BANDS[2]+'F',color='b')
                if True: #plot error
                    er = 0.1
                    ax0.errorbar(x=df.index,y=df[BANDS[0]+'F']*mi,yerr=df[BANDS[0]+'F']*mi*er,color='b', ecolor='b', alpha = 0.6)
                    ax1.errorbar(x=df.index,y=df[BANDS[1]+'F']*mi,yerr=df[BANDS[1]+'F']*mi*er,color='b', ecolor='b', alpha = 0.6)
                    ax2.errorbar(x=df.index,y=df[BANDS[2]+'F']*mi,yerr=df[BANDS[2]+'F']*mi*er,color='b', ecolor='b', alpha = 0.6)
                # ratios
                ax3.plot(df.index, df[RATIOS[0]+'F'],label=RATIOS[0]+'F',color='b')
                ax4.plot(df.index, df[RATIOS[1]+'F'],label=RATIOS[1]+'F',color='b')
                if True: #plot error
                    er = 0.1
                    ax3.errorbar(x=df.index,y=df[RATIOS[0]+'F'],yerr=df[RATIOS[0]+'F']*er,color='b', ecolor='b', alpha = 0.6)
                    ax4.errorbar(x=df.index,y=df[RATIOS[1]+'F'],yerr=df[RATIOS[1]+'F']*er,color='b', ecolor='b', alpha = 0.6)
                #
                ax5.errorbar(x=df.index,y=df['Qm'],yerr=df['Qs']/2,label='Q: mean gas flux',color='r', ecolor='r')
                ax6.errorbar(x=df.index,y=df['Dm'],yerr=df['Ds']/2,label='D: gas pocket thickness',color='g', ecolor='g')
                ax7.errorbar(x=df.index,y=df['Nm'],yerr=df['Ns']/2,label='N: Number of impulses',color='m', ecolor='m')
            #
            if fit:
                if len(BANDS)==1:
                    f_l=[]
                    for i in range(len(df)):
                        DP, DP2, U_z, V_z = fwd_cav_Ps_Uz_Vz(D=df['Dm'][i], Q=df['Qm'][i], N=df['Nm'][i])
                        f_l.append(np.mean(np.abs(V_z)))
                    mf=np.sum(sqrt((f_l*mi-df['rsamF']*mi)**2))/len(df)
                    ax0.plot(df.index, f_l*mi,'k--',label='est_RSAM (L2: '+str(round(mf,1))+')')
                    #
                    ax0.legend(loc=1)
                    ax1.legend(loc=1)
                    ax2.legend(loc=1)
                    ax0.set_ylabel('RSAM')
                    ax1.set_ylabel('Q [kg/s]')
                    ax2.set_ylabel('D [m]')
                    ax3.set_ylabel('N')
                if len(BANDS+RATIOS)==3:
                    f_l_0,f_l_1,f_l_2,f_l_p=[],[],[],[]
                    for i in range(len(df)):
                        #DP, DP2, U_z, V_z = fwd_cav_Ps_Uz_Vz(D=df['Dm'][i], Q=df['Qm'][i], N=df['Nm'][i])
                        #_datas= Vz_filters(V_z)
                        #
                        if (lookup_table):# and not Pressure):
                            #
                            D=df['Dm'][i]
                            Q=df['Qm'][i]
                            N=df['Nm'][i]
                            p=np.array([D,Q,N])    # random parameter vector
                            est = [rgi_rsam(p)[0],rgi_mf(p)[0],rgi_hf(p)[0]] # look up simulation result
                            f_l_0.append(est[0])
                            f_l_1.append(est[1])
                            f_l_2.append(est[2])
                            if Pressure:
                                f_l_p.append(rgi_dp(p)[0])
                        else:
                            ## estimate square function of clay content given pars [z1,z2,%] 
                            if Pressure:
                                DP, DP2, U_z, V_z = fwd_cav_Ps_Uz_Vz(D=df['Dm'][i], Q=df['Qm'][i], N=df['Nm'][i], Pressure=True)
                            else:
                                DP, DP2, U_z, V_z = fwd_cav_Ps_Uz_Vz(D=df['Dm'][i], Q=df['Qm'][i], N=df['Nm'][i])
                            #est = np.mean(np.abs(V_z)*1e6)
                            _datas= Vz_filters(V_z)
                            #est = [np.mean(np.abs(_d)) for _d in _datas] 
                            #for _d in _datas:
                            f_l_0.append(np.mean(np.abs(_datas[0])))
                            f_l_1.append(np.mean(np.abs(_datas[1])))
                            f_l_2.append(np.mean(np.abs(_datas[2])))
                            if Pressure:
                                f_l_p.append(np.mean(np.real(DP)))
                            #
                    f_l_0=np.asarray(f_l_0)
                    f_l_1=np.asarray(f_l_1)
                    f_l_2=np.asarray(f_l_2)
                    #
                    mf=np.sum(sqrt((f_l_0*mi-df['rsamF']*mi)**2))/len(df)
                    ax0.plot(df.index, f_l_0*mi,'k--',label='est_RSAM (L2: '+str(round(mf,1))+')')

                    mf=np.sum(sqrt((f_l_1*mi-df['hfF']*mi)**2))/len(df)
                    ax2.plot(df.index, f_l_1*mi,'k--',label='est_MF (L2: '+str(round(mf,1))+')')

                    mf=np.sum(sqrt((f_l_2*mi-df['hfF']*mi)**2))/len(df)
                    ax4.plot(df.index, f_l_2*mi,'k--',label='est_HF (L2: '+str(round(mf,1))+')')

                    if Pressure:
                        #mf=np.sum(sqrt((f_l_2-df['hfF'])**2))/len(df)
                        ax6.plot(df.index, np.array(f_l_p),'k--',label='Pressure')
                    #
                    ax0.legend(loc=1)
                    ax1.legend(loc=1)
                    ax2.legend(loc=1)
                    ax3.legend(loc=1)
                    ax4.legend(loc=1)
                    ax5.legend(loc=1)
                    if mi != 1.:
                        ax0.set_ylabel('RSAM [1e6 m/s]')
                        ax2.set_ylabel('MF [1e6 m/s]')
                        ax4.set_ylabel('HF [1e6 m/s]')
                    ax1.set_ylabel('Q [kg/s]')
                    ax3.set_ylabel('D [m]')
                    ax5.set_ylabel('N')
                    if Pressure:
                        ax6.set_ylabel('P [Pa]')

                    ax0.set_yscale('log')
                    ax2.set_yscale('log')
                    ax4.set_yscale('log')
                    if Pressure:
                        ax6.set_yscale('log')
                if len(BANDS+RATIOS)==5:
                    f_l_0,f_l_1,f_l_2,f_l_3,f_l_4,f_l_p=[],[],[],[],[],[]
                    for i in range(len(df)):
                        #DP, DP2, U_z, V_z = fwd_cav_Ps_Uz_Vz(D=df['Dm'][i], Q=df['Qm'][i], N=df['Nm'][i])
                        #_datas= Vz_filters(V_z)
                        #
                        if (lookup_table):# and not Pressure):
                            #
                            D=df['Dm'][i]
                            Q=df['Qm'][i]
                            N=df['Nm'][i]
                            p=np.array([D,Q,N])    # random parameter vector
                            est = [rgi_rsam(p)[0],rgi_mf(p)[0],rgi_hf(p)[0],rgi_rmar(p)[0],rgi_dsar(p)[0]] # look up simulation result
                            f_l_0.append(est[0])
                            f_l_1.append(est[1])
                            f_l_2.append(est[2])
                            f_l_3.append(est[3])
                            f_l_4.append(est[4])
                            if Pressure:
                                f_l_p.append(rgi_dp(p)[0])
                        else:
                            ## estimate square function of clay content given pars [z1,z2,%] 
                            if Pressure:
                                DP, DP2, U_z, V_z = fwd_cav_Ps_Uz_Vz(D=df['Dm'][i], Q=df['Qm'][i], N=df['Nm'][i], Pressure=True)
                            else:
                                DP, DP2, U_z, V_z = fwd_cav_Ps_Uz_Vz(D=df['Dm'][i], Q=df['Qm'][i], N=df['Nm'][i])
                            #est = np.mean(np.abs(V_z)*1e6)
                            _datas= Vz_filters(V_z)
                            if RATIOS:
                                _datas_r= calc_ratios(U_z)
                            #est = [np.mean(np.abs(_d)) for _d in _datas] 
                            #for _d in _datas:
                            f_l_0.append(np.mean(np.abs(_datas[0])))
                            f_l_1.append(np.mean(np.abs(_datas[1])))
                            f_l_2.append(np.mean(np.abs(_datas[2])))
                            f_l_3.append(np.mean(np.abs(_datas_r[0])))
                            f_l_4.append(np.mean(np.abs(_datas_r[1])))
                            if Pressure:
                                f_l_p.append(np.mean(np.real(DP2)))
                            #
                    
                    f_l_0=np.asarray(f_l_0) 
                    f_l_1=np.asarray(f_l_1)
                    f_l_2=np.asarray(f_l_2)
                    f_l_3=np.asarray(f_l_3)
                    f_l_4=np.asarray(f_l_4)
                    #
                    #
                    mfit=np.sum(sqrt((f_l_0*mi-df['rsamF']*mi)**2))/len(df)
                    ax0.plot(df.index, f_l_0*mi,'k--',label='est_RSAM (L2: '+str(round(mfit,1))+')')

                    mfit=np.sum(sqrt((f_l_1*mi-df['hfF']*mi)**2))/len(df)
                    ax1.plot(df.index, f_l_1*mi,'k--',label='est_MF (L2: '+str(round(mfit,1))+')')

                    mfit=np.sum(sqrt((f_l_2*mi-df['hfF']*mi)**2))/len(df)
                    ax2.plot(df.index, f_l_2*mi,'k--',label='est_HF (L2: '+str(round(mfit,1))+')')

                    if RATIOS:
                        #dataI = cumtrapz(data, dx=1./100, initial=0)
                        #f_l_3=cumtrapz(f_l_0/f_l_1, dx=1, initial=0) 
                        #f_l_4 = f_l_4[:-1]-f_l_4[1:]
                        mfit=np.sum(sqrt((f_l_3-df['rmarF'])**2))/len(df)
                        ax3.plot(df.index, f_l_3,'k--',label='est_RMAR (L2: '+str(round(mfit,1))+')')
                        #mfit=np.sum(sqrt((f_l_4-df['dsarF'])**2))/len(df)
                        ax4.plot(df.index, f_l_4,'k--',label='est_DSAR (L2: '+str(round(mfit,1))+')')
                    if Pressure:
                        #mf=np.sum(sqrt((f_l_2-df['hfF'])**2))/len(df)
                        ax8.plot(df.index, np.array(f_l_p),'k--',label='Pressure')
                    #
                    ax0.legend(loc=1)
                    ax1.legend(loc=1)
                    ax2.legend(loc=1)
                    ax3.legend(loc=1)
                    ax4.legend(loc=1)
                    ax5.legend(loc=1)
                    ax6.legend(loc=1)
                    ax7.legend(loc=1)
                    ax8.legend(loc=1)
                    if mi != 1.:
                        ax0.set_ylabel('RSAM [1e6 m/s]')
                        ax1.set_ylabel('MF [1e6 m/s]')
                        ax2.set_ylabel('HF [1e6 m/s]')
                        ax3.set_ylabel('RMAR')
                        ax4.set_ylabel('DSAR')
                    ax5.set_ylabel('Q [kg/s]')
                    ax6.set_ylabel('D [m]')
                    ax7.set_ylabel('N')
                    if Pressure:
                        ax8.set_ylabel('P [Pa]')
                    
                    ax0.set_yscale('log')
                    ax1.set_yscale('log')
                    ax2.set_yscale('log')
                    if True:
                        ax3.set_yscale('log')
                        ax4.set_yscale('log')
                        ax3.set_ylim([1,100])
                        ax4.set_ylim([1,100])

                    if Pressure:
                        ax8.set_yscale('log')
            if plot_erup:
                for ax in [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
                    ax.axvline(x=datetimeify(plot_erup),linewidth=4, color='gray', alpha = 0.6)
    else:
        if len(BANDS) == 3:
    
            if Pressure:
                fig, (ax0, ax2, ax4, ax1, ax3, ax5, ax6) = plt.subplots(7, 1, figsize=(18, 12))
            else:
                fig, (ax0, ax2, ax4, ax1, ax3, ax5) = plt.subplots(6, 1, figsize=(16, 12))
            f_l_0,f_l_1,f_l_2,f_l_p=[],[],[],[]

            #rsam
            if FILTERED_DATA:
                ax0.plot(df.index, df[BANDS[0]+'F']*mi,label=BANDS[0]+'F',color='b')
                ax2.plot(df.index, df[BANDS[1]+'F']*mi,label=BANDS[1]+'F',color='b')
                ax4.plot(df.index, df[BANDS[2]+'F']*mi,label=BANDS[2]+'F',color='b')

                ax1.errorbar(x=df.index,y=df['Qm'],yerr=df['Qs']/2,label='Q: mean gas flux',color='r', ecolor='r')
                ax3.errorbar(x=df.index,y=df['Dm'],yerr=df['Ds']/2,label='D: gas pocket thickness',color='g', ecolor='g')
                ax5.errorbar(x=df.index,y=df['Nm'],yerr=df['Ns']/2,label='N: Nomber of impulses',color='m', ecolor='m')

            for i in range(len(df)):
                #DP, DP2, U_z, V_z = fwd_cav_Ps_Uz_Vz(D=df['Dm'][i], Q=df['Qm'][i], N=df['Nm'][i])
                #_datas= Vz_filters(V_z)
                #
                if (lookup_table):# and not Pressure):
                    #
                    D=df['Dm'][i]
                    Q=df['Qm'][i]
                    N=df['Nm'][i]
                    p=np.array([D,Q,N])    # random parameter vector
                    est = [rgi_rsam(p)[0],rgi_mf(p)[0],rgi_hf(p)[0]] # look up simulation result
                    f_l_0.append(est[0])
                    f_l_1.append(est[1])
                    f_l_2.append(est[2])
                    if Pressure:
                        f_l_p.append(rgi_dp(p)[0])
                else:
                    ## estimate square function of clay content given pars [z1,z2,%] 
                    if Pressure:
                        DP, DP2, U_z, V_z = fwd_cav_Ps_Uz_Vz(D=df['Dm'][i], Q=df['Qm'][i], N=df['Nm'][i], Pressure=True)
                    else:
                        DP, DP2, U_z, V_z = fwd_cav_Ps_Uz_Vz(D=df['Dm'][i], Q=df['Qm'][i], N=df['Nm'][i])
                    #est = np.mean(np.abs(V_z)*1e6)
                    _datas= Vz_filters(V_z)
                    #est = [np.mean(np.abs(_d)) for _d in _datas] 
                    #for _d in _datas:
                    f_l_0.append(np.mean(np.abs(_datas[0])))
                    f_l_1.append(np.mean(np.abs(_datas[1])))
                    f_l_2.append(np.mean(np.abs(_datas[2])))
                    if Pressure:
                        f_l_p.append(np.mean(np.real(DP)))
                    #
            f_l_0=np.asarray(f_l_0)
            f_l_1=np.asarray(f_l_1)
            f_l_2=np.asarray(f_l_2)
            #
            mf=np.sum(sqrt((f_l_0*mi-df['rsamF']*mi)**2))/len(df)
            ax0.plot(df.index, f_l_0*mi,'k--',label='est_RSAM (L2: '+str(round(mf,1))+')')

            mf=np.sum(sqrt((f_l_1*mi-df['hfF']*mi)**2))/len(df)
            ax2.plot(df.index, f_l_1*mi,'k--',label='est_MF (L2: '+str(round(mf,1))+')')

            mf=np.sum(sqrt((f_l_2*mi-df['hfF']*mi)**2))/len(df)
            ax4.plot(df.index, f_l_2*mi,'k--',label='est_HF (L2: '+str(round(mf,1))+')')

            if Pressure:
                #mf=np.sum(sqrt((f_l_2-df['hfF'])**2))/len(df)
                ax6.plot(df.index, np.array(f_l_p),'k--',label='Pressure')
            #
            ax0.legend(loc=1)
            ax1.legend(loc=1)
            ax2.legend(loc=1)
            ax3.legend(loc=1)
            ax4.legend(loc=1)
            ax5.legend(loc=1)
            if mi != 1.:
                ax0.set_ylabel('RSAM [1e6 m/s]')
                ax2.set_ylabel('MF [1e6 m/s]')
                ax4.set_ylabel('HF [1e6 m/s]')
            ax1.set_ylabel('Q [kg/s]')
            ax3.set_ylabel('D [m]')
            ax5.set_ylabel('N')
            if Pressure:
                ax6.set_ylabel('P [Pa]')
    if _dir:
        plt.savefig(_dir+os.sep+'pars_fit.png', dpi=300)
    else:
        plt.savefig('pars_fit.png', dpi=300)
def re_read_chains(_dir=None):
    '''
    re read chains to compute percentiles and add to df pars

    _dir: directory to load and save files (None for current directory)
    '''
    if _dir:
        fl_df=_dir+os.sep+'df_pars.csv'
    else:
        fl_df='df_pars.csv'
    df = pd.read_csv(fl_df, index_col= 'time', infer_datetime_format=True)
    # Converting the index as date
    df.index = pd.to_datetime(df.index)
    #
    pars = ['D','Q','N']
    for k,p in enumerate(pars):
        p10_l, p20_l, p30_l, p40_l, p50_l, p60_l, p70_l, p80_l, p90_l = [],[],[],[],[],[],[],[],[]
        for i in range(len(df)):

            # read chain
            if _dir:
                chain = np.genfromtxt(_dir+os.sep+'chains/chain'+str(i)+'.dat')
            else:
                chain = np.genfromtxt('./chains/chain'+str(i)+'.dat')
            npar = int(chain.shape[1] - 3)
            #pars = [None] * npar
            #for j in [3]:#range(npar+2): # 
            #L = chain[:,j]#[:,j+2]
            _L = chain[:,k+2]
            p10_l.append(np.percentile(_L,10))
            p20_l.append(np.percentile(_L,20))
            p30_l.append(np.percentile(_L,30))
            p40_l.append(np.percentile(_L,40))
            p50_l.append(np.percentile(_L,50))
            p60_l.append(np.percentile(_L,60))
            p70_l.append(np.percentile(_L,70))
            p80_l.append(np.percentile(_L,80))
            p90_l.append(np.percentile(_L,90))
        df[p+'_p10'] = [round(i,3) for i in p10_l]
        df[p+'_p20'] = [round(i,3) for i in p20_l]
        df[p+'_p30'] = [round(i,3) for i in p30_l]
        df[p+'_p40'] = [round(i,3) for i in p40_l]
        df[p+'_p50'] = [round(i,3) for i in p50_l]
        df[p+'_p60'] = [round(i,3) for i in p60_l]
        df[p+'_p70'] = [round(i,3) for i in p70_l]
        df[p+'_p80'] = [round(i,3) for i in p80_l]
        df[p+'_p90'] = [round(i,3) for i in p90_l]
    #
    if _dir:
        df.to_csv(_dir+os.sep+'df_pars_pers.csv')
    else:
        df.to_csv('df_pars_pers.csv')
def plot_pars_pers(fit=True, _dir= None):
    '''
    _dir: directory to load and save files (None for current directory)
    '''
    if _dir:
        df = pd.read_csv(_dir+os.sep+'df_pars_pers.csv', index_col= 'time', infer_datetime_format=True)
    else:
        df = pd.read_csv('df_pars_pers.csv', index_col= 'time', infer_datetime_format=True)    
    # Converting the index as date
    df.index = pd.to_datetime(df.index)
    #
    mi=1.e6 # 1.
    #
    npar = int((df.shape[1]-1)/2)
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(14, 7))
    #rsam
    ax0.plot(df.index, df['rsamF']*mi,label='RSAM',color='b')
    ax1.errorbar(x=df.index,y=df['Qm'],yerr=df['Qs']/2,label='Q: mean gas flux (mean and std)',color='r', ecolor='r')
    ax2.errorbar(x=df.index,y=df['Dm'],yerr=df['Ds']/2,label='D: gas pocket thickness (mean and std)',color='g', ecolor='g')
    ax3.errorbar(x=df.index,y=df['Nm'],yerr=df['Ns']/2,label='N: Nomber of impulses (mean and std)',color='m', ecolor='m')
    #
    ax1.plot(df.index, df['Q_p10']*mi,'--',label='p10',color='r')
    ax1.plot(df.index, df['Q_p50']*mi,':',label='p50',color='r')
    ax1.plot(df.index, df['Q_p90']*mi,'-.',label='p90',color='r')
    ax2.plot(df.index, df['D_p10']*mi,'--',label='p10',color='g')
    ax2.plot(df.index, df['D_p50']*mi,':',label='p50',color='g')
    ax2.plot(df.index, df['D_p90']*mi,'-.',label='p90',color='g')
    ax3.plot(df.index, df['N_p10']*mi,'--',label='p10',color='m')
    ax3.plot(df.index, df['N_p50']*mi,':',label='p50',color='m')
    ax3.plot(df.index, df['N_p90']*mi,'-.',label='p90',color='m')
    #
    if True:#fit:
        f_l=[]
        for i in range(len(df)):
            #DP, DP2, U_z, V_z = fwd_cav_Ps_Uz_Vz(D=df['D_p50'][i], Q=df['Q_p50'][i], N=int(df['N_p50'][i]))
            #f_l.append(np.mean(np.abs(V_z)))
            #
            if lookup_table:
                D=df['Dm'][i]
                Q=df['Qm'][i]
                N=df['Nm'][i]
                p=np.array([D,Q,N])    # random parameter vector
                est = [rgi_rsam(p)[0],rgi_mf(p)[0],rgi_hf(p)[0]] # look up simulation result
                f_l.append(est[0])
                #f_l_1.append(est[1])
                #f_l_2.append(est[2])
            else:
                ## estimate square function of clay content given pars [z1,z2,%] 
                DP, DP2, U_z, V_z = fwd_cav_Ps_Uz_Vz(D=df['Dm'][i], Q=df['Qm'][i], N=df['Nm'][i])
                #est = np.mean(np.abs(V_z)*1e6)
                _datas= Vz_filters(V_z)
                #est = [np.mean(np.abs(_d)) for _d in _datas] 
                #for _d in _datas:
                f_l.append(np.mean(np.abs(_datas[0])))
                #f_l_1.append(np.mean(np.abs(_datas[1])))
                #f_l_2.append(np.mean(np.abs(_datas[2])))
                #
        f_l=np.array(f_l)
        mf=np.sum(sqrt((f_l*mi-df['rsamF']*mi)**2))/len(df)
        ax0.plot(df.index, f_l*mi,'k--',label='est_RSAM (L2: '+str(round(mf,1))+')')
    ax0.legend(loc=1)
    ax1.legend(loc=1)
    ax2.legend(loc=1)
    ax3.legend(loc=1)
    ax0.set_ylabel('RSAM')
    ax1.set_ylabel('Q [kg/s]')
    ax2.set_ylabel('D [m]')
    ax3.set_ylabel('N')

    ax0.set_yscale('log')
    if _dir:
        plt.savefig(_dir+os.sep+'pars_fit_pers.png', dpi=300)
    else:
        plt.savefig('pars_fit_pers.png', dpi=300)   
##########################################################
def main():
    #plot_pars(_dir=r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\mcmc_rsam\mcmc_Q_rsam_3D\test')
    #plot_results_mcmc(fl='./chains/chain1.dat')
    #plot_results_mcmc(fl='./mcmc_inv/2006-10-04/chains/chain1.dat')
    #asdf
    ##
    if False : # run mcmc for one tailored time
        import time as _time
        st = _time.time()
        ##
        #ex1 (1.5 days, 1 s/hour, 8 hours run)
        #ti='2016-11-13 12:00:00'
        #tf='2016-11-15 00:00:00'
        #ex2 (2 days, .5 s/hour, 8 hours run)
        ti='2016-11-12 12:00:00'
        tf='2016-11-14 12:00:00'
        #ex3 (.5 days, 1 s/hour,  hours run)
        #ti='2016-11-13 12:00:00'
        #tf='2016-11-14 00:00:00'
        #ex4 (.5 days, 1 s/hour,  hours run)
        ti='2016-11-12 23:00:00'
        ztf='2016-11-13 23:00:00'
        df = _load_tremor_data(ti,tf, downsample=3)
        #
        Qm_l,Qs_l,Dm_l,Ds_l,Nm_l,Ns_l=[],[],[],[],[],[]
        for i in range(len(df)):
            run_mcmc(df['rsamF'][i], walk_jump = 500)
            pars=read_chain_pars()
            #
            Qm_l.append(pars[1][0])
            Qs_l.append(pars[1][1])
            Dm_l.append(pars[0][0])
            Ds_l.append(pars[0][1])
            Nm_l.append(pars[2][0])
            Ns_l.append(pars[2][1])
            os.replace("chain.dat", "./chains/chain"+str(i)+".dat")
            #
        df['Dm'] = [round(i,3) for i in Dm_l]
        df['Ds'] = [round(i,3) for i in Ds_l]
        df['Qm'] = [int(i) for i in Qm_l]
        df['Qs'] = [int(i) for i in Qs_l]
        df['Nm'] = [int(i) for i in Nm_l]
        df['Ns'] = [int(i) for i in Ns_l]
        #
        df.to_csv('df_pars.csv')
        ###########################
        # get the end time
        et = _time.time()
        # get the execution time
        elapsed_time = et - st
        print('Execution time:', elapsed_time, 'seconds')
        ##############################
    if False: # basic plots for one inversion 
        pass
        #plot_pars()
        fl=r'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\mcmc_rsam\\mcmc_Q_rsam_3D\\ex5\\mcmc_inv\\2006-10-04\\chains\\chain14.dat'
        #plot_results_mcmc(fl=fl) 
        _dir=r'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\mcmc_rsam\\mcmc_Q_rsam_3D\\test\\mcmc_inv\\FWVZ\\2016-11-13'
        _dir=r'C:\\Users\\aar135\\codes_local_disk\\volc_forecast_tl\\mcmc_rsam\\mcmc_Q_rsam_3D\\test\\mcmc_inv\\FWVZ\\2016-11-13'
        plot_pars(_dir=_dir, Pressure=True)
        #re_read_chains(fl_df='df_pars.csv')
        #plot_pars_pers()
    if True: # run inversion for several eruption in Ruapehu and Whakaari (for short periods; ~ month)
        sta='GOD'#FWVZ'#'FWVZ'
        look_back = 5#75 # days before eruption
        look_front = 1#21#1.5 # days after eruption
        # station (volcano) and list of eruption times 
        import time as _time
        if sta == 'FWVZ':
            tes = ['2006-10-04 09:20:00',
                    '2016-11-13 11:00:00',
                    '2007-11-06 22:10:00',
                    '2009-07-13 06:30:00',
                    '2010-09-03 16:30:00',
                    '2021-03-04 13:20:00'
                    ]
            tes = ['2016-11-13 11:00:00']
        if sta == 'WIZ':
            tes = ['2012-08-04 16:52:00',
                    '2013-08-19 22:23:00',
                    '2013-10-11 07:09:00',
                    '2016-04-27 09:37:00',
                    '2019-12-09 01:11:00']
            #tes = ['2016-04-27 09:37:00']
        if sta == 'GOD':
            tes = ['2010-03-20 12:00:00',
                    '2010-04-14 12:00:00']
            tes = ['2010-04-14 12:00:00']
        if sta == 'ONTA':
            tes = ['2014-09-27 12:00:00']##11:52:00']
        #tes = ['2008-12-31 23:50:00']
        # check directory structure
        newpath = r'.'+os.sep+'mcmc_inv' 
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        newpath = r'.'+os.sep+'mcmc_inv'+os.sep+sta 
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        for te in tes:
            newpath = r'.'+os.sep+'mcmc_inv'+os.sep+sta+os.sep+''+te.split(' ')[0] 
            if not os.path.exists(newpath):
                os.makedirs(newpath)
        # loop over tes to run inversion 
        for te in tes: 
            print('Running McMC: '+te)
            st = _time.time()
            # time period
            _te=datetimeify(te)
            ti=_te-look_back*_DAY
            tf=_te+look_front*_DAY
            # load rsam data 
            if sta == 'WIZ':
                fl= 'C:'+os.sep+'Users'+os.sep+'aar135'+os.sep+'codes_local_disk'+os.sep+'volc_forecast_tl'+os.sep+'volc_forecast_tl'+os.sep+'data'+os.sep+'WIZ_tremor_data.csv'
                fl= 'E:'+os.sep+'data_bkp'+os.sep+'WIZ_tremor_data.csv'
            if sta == 'FWVZ':
                fl= 'C:'+os.sep+'Users'+os.sep+'aar135'+os.sep+'codes_local_disk'+os.sep+'volc_forecast_tl'+os.sep+'volc_forecast_tl'+os.sep+'data'+os.sep+'FWVZ_tremor_data.csv'
            if sta == 'PVV':
                fl= 'C:'+os.sep+'Users'+os.sep+'aar135'+os.sep+'codes_local_disk'+os.sep+'volc_forecast_tl'+os.sep+'volc_forecast_tl'+os.sep+'data'+os.sep+'PVV_tremor_data.csv'
            if sta == 'GOD':
                fl= 'C:'+os.sep+'Users'+os.sep+'aar135'+os.sep+'codes_local_disk'+os.sep+'volc_forecast_tl'+os.sep+'volc_forecast_tl'+os.sep+'data'+os.sep+'GOD_tremor_data.csv'
                #fl= 'E:'+os.sep+'data_bkp'+os.sep+'WIZ_tremor_data.csv'
            if sta == 'ONTA':
                fl= 'C:'+os.sep+'Users'+os.sep+'aar135'+os.sep+'codes_local_disk'+os.sep+'volc_forecast_tl'+os.sep+'volc_forecast_tl'+os.sep+'data'+os.sep+'ONTA_tremor_data.csv'
            #
            df = _load_tremor_data(ti,tf, downsample=1, fl = fl)
            #
            if sta == 'ONTA':
                for column in df:
                    if column in ['rsam','rsamF','mf','mfF','hf','hfF']:
                        df[column] = df[column].apply(lambda x: x/1e9)
            if False:
                f, axes = plt.subplots(len(BANDS+RATIOS), 1)
                for i,_b in enumerate(BANDS+RATIOS):
                    if FILTERED_DATA:
                        axes[i].plot(df[_b+'F'], label=_b)
                    else:
                        axes[i].plot(df[_b], label=_b)
                    axes[i].legend(loc=1)
                plt.show()
                asdf
            #    
            print('Points to invert: '+str(len(df)))
            # run mcmc
            run_mcmc_aux(df)
            # get the end time
            et = _time.time()
            # get the execution time
            elapsed_time = et - st
            print('Execution time:', elapsed_time, 'seconds\n')
            # plots
            _dir=None#r'C:\Users\aar135\codes_local_disk\volc_forecast_tl\mcmc_rsam\mcmc_Q_rsam_3D\ex6\mcmc_inv\2006-10-04'
            plot_pars(_dir=_dir, Pressure=True, plot_erup=te)
            #re_read_chains(fl_df='df_pars.csv')
            plot_pars_pers(_dir=_dir)
            #plot_results_mcmc(fl='./chains/chain0.dat')
            # move files to directory 
            #os.rename('.'+os.sep+'chains', r'.'+os.sep+'mcmc_inv'+os.sep+sta+os.sep+''+te.split(' ')[0]+''+os.sep+'chains')
            #os.rename('.'+os.sep+'df_pars.csv', r'.'+os.sep+'mcmc_inv'+os.sep+sta+os.sep+''+te.split(' ')[0]+''+os.sep+'df_pars.csv')
            #os.rename('.'+os.sep+'df_pars_pers.csv', r'.'+os.sep+'mcmc_inv'+os.sep+sta+os.sep+''+te.split(' ')[0]+''+os.sep+'df_pars_pers.csv')
            #os.rename('.'+os.sep+'pars_fit.png', r'.'+os.sep+'mcmc_inv'+os.sep+sta+os.sep+''+te.split(' ')[0]+''+os.sep+'pars_fit.png')
            #os.rename('.'+os.sep+'pars_fit_pers.png', r'.'+os.sep+'mcmc_inv'+os.sep+sta+os.sep+''+te.split(' ')[0]+''+os.sep+'pars_fit_pers.png') 
            #           
    if False: # run inversion for extended periods in Ruapehu and Whakaari (for long periods; ~ years)
        sta='FWVZ'#'FWVZ'
        #look_back = .5 # days before eruption
        #look_front = 1.5 # days after eruption
        # station (volcano) and list of eruption times 
        import time as _time
        if sta == 'FWVZ':
            period = ['2011-01-01 00:00:00','2021-00-00 00:00:00']
        if sta == 'WIZ':
            period = ['2019-01-01','2019-12-31']
        print('Full inversion period. From '+period[0]+' to '+period[1]+'\n')
        #
        periods = _period_into_months(period[0],period[1])
        # loop over periods
        for i,p in enumerate(periods):
            newpath = r'.'+os.sep+'mcmc_inv_long_'+period[0].split('-')[0]+os.sep+''+str(p[0]).split(' ')[0] 
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            # directory to save file (inside of newpath, or by default in current directory)
            _dir = None#newpath 
            #
            st = _time.time()
            # # loop over tes to run inversion 
            ti=p[0]
            tf=p[1]
            if sta == 'WIZ':
                fl= 'C:'+os.sep+'Users'+os.sep+'aar135'+os.sep+'codes_local_disk'+os.sep+'volc_forecast_tl'+os.sep+'volc_forecast_tl'+os.sep+'data'+os.sep+'WIZ_tremor_data.csv'
                fl= '/media/eruption_forecasting/eruptions/data'+os.sep+'WIZ_tremor_data.csv'
                fl= '..'+os.sep+'..'+os.sep+'data'+os.sep+'WIZ_tremor_data.csv'
            if sta == 'FWVZ':
                fl= 'C:'+os.sep+'Users'+os.sep+'aar135'+os.sep+'codes_local_disk'+os.sep+'volc_forecast_tl'+os.sep+'volc_forecast_tl'+os.sep+'data'+os.sep+'FWVZ_tremor_data.csv'
                fl= '..'+os.sep+'..'+os.sep+'data'+os.sep+'FWVZ_tremor_data.csv'
            # import data and select downsampling
            df = _load_tremor_data(ti,tf, downsample=1, fl = fl)
            print('Interting from '+str(p[0]).split(' ')[0]+' to '+str(p[1]).split(' ')[0])
            print('Points to invert: '+str(len(df)))
            # run mcmc
            run_mcmc_aux(df, _dir)
            # get the end time
            et = _time.time()
            # get the execution time
            elapsed_time = et - st
            if i == 0:
                print('Execution time:', elapsed_time, 'seconds\n')
            # plots
            plot_pars(_dir=_dir)
            #plot_results_mcmc(fl='./chains/chain30.dat') 
            #re_read_chains(fl_df='df_pars.csv')
            plot_pars_pers(_dir=_dir)
            # move files to directory 
            if _dir is None:
                os.rename('.'+os.sep+'chains', newpath+os.sep+'chains')
                os.rename('.'+os.sep+'df_pars.csv', newpath+os.sep+'df_pars.csv')
                os.rename('.'+os.sep+'df_pars_pers.csv', newpath+os.sep+'df_pars_pers.csv')
                os.rename('.'+os.sep+'pars_fit.png', newpath+os.sep+'pars_fit.png')
                os.rename('.'+os.sep+'pars_fit_pers.png', newpath+os.sep+'pars_fit_pers.png')
    if False: # results analysis 
        if False: # scatter plots or parameters to check correlations 
            sta='FWVZ'#'FWVZ'
            # station (volcano) and list of eruption times 
            if sta == 'FWVZ':
                tes = ['2006-10-04 09:20:00',
                        '2007-11-06 22:10:00',
                        '2009-07-13 06:30:00',
                        '2010-09-03 16:30:00',
                        '2016-11-13 11:00:00',
                        '2021-03-04 13:20:00'
                        ]
            if sta == 'WIZ':
                tes = ['2012-08-04 16:52:00',
                        '2013-08-19 22:23:00',
                        '2013-10-11 07:09:00',
                        '2016-04-27 09:37:00',
                        '2019-12-09 01:11:00']
            dir_path = r'.'+os.sep+'ex6'+os.sep+'mcmc_inv'+os.sep
            #dir_path = r'.'+os.sep+'ex7_wiz'+os.sep+'mcmc_inv'+os.sep
            # collect data 
            pars = ['time','Q_p50','D_p50','N_p50']
            for i,te in enumerate(tes):
                newpath = dir_path+''+te.split(' ')[0] 
                df = pd.read_csv(newpath+os.sep+'df_pars_pers.csv', infer_datetime_format=True, usecols=pars)
                df['time'] = pd.to_datetime(df['time'])
                df = df.set_index('time')
                # add marker
                df['erup'] = df.index>=te 
                # concat
                if i == 0:
                    df_f = df
                if i>0:
                    df_f=pd.concat([df_f,df])
            # library & dataset
            #df = sns.load_dataset('iris')
            # with regression
            #sns.pairplot(df, kind="reg")
            #plt.show()
            # without regression
            #sca_plot = sns.pairplot(df_f, kind="scatter")
            sca_plot = sns.pairplot(df_f, diag_kind="kde", hue='erup', hue_order=[True, False], diag_kws=dict(shade=True, vertical=False) )
            plt.savefig(dir_path+os.sep+"scatter_corr.png")
            #
        if True: #  plot median of events toeghter for all events (in one volcano) 
            sta='WIZ'#'FWVZ'
            # station (volcano) and list of eruption times 
            if sta == 'FWVZ':
                tes = ['2006-10-04 09:20:00',
                        '2007-11-06 22:10:00',
                        '2009-07-13 06:30:00',
                        '2010-09-03 16:30:00',
                        '2016-11-13 11:00:00',
                        '2021-03-04 13:20:00'
                        ]
            if sta == 'WIZ':
                tes = ['2012-08-04 16:52:00',
                        '2013-08-19 22:23:00',
                        '2013-10-11 07:09:00',
                        '2016-04-27 09:37:00',
                        '2019-12-09 01:11:00']
            dir_path = r'.'+os.sep+'ex7_wiz'+os.sep+'mcmc_inv'+os.sep
            #dir_path = r'.'+os.sep+'ex7_wiz'+os.sep+'mcmc_inv'+os.sep
            # collect data 
            pars = ['time','rsamF','Q_p50','D_p50','N_p50']
            labels = ['D','Q','N']
            from random import randint
            colors = []
            [colors.append('#%06X' % randint(0, 0xFFFFFF)) for i in range(len(tes))]
            #npar = int((df.shape[1]-1)/2)
            fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(12, 10))
            for i,te in enumerate(tes):
                newpath = dir_path+''+te.split(' ')[0] 
                df = pd.read_csv(newpath+os.sep+'df_pars_pers.csv', usecols=pars)
                #df['time'] = pd.to_datetime(df['time'])
                #df = df.set_index('time')
                #rsam
                ax0.plot(df.index,df['rsamF'],'-',label=te.split(' ')[0],color=colors[i])
                ax1.plot(df.index,df['Q_p50'],'-',label=te.split(' ')[0],color=colors[i])
                ax2.plot(df.index,df['D_p50'],'-',label=te.split(' ')[0],color=colors[i])
                ax3.plot(df.index,df['N_p50'],'-',label=te.split(' ')[0],color=colors[i])

            ax0.legend(loc=1)
            ax1.legend(loc=1)
            ax2.legend(loc=1)
            ax0.set_ylabel('RSAM')
            ax1.set_ylabel('Q [kg/s]')
            ax2.set_ylabel('D [m]')
            ax3.set_ylabel('N')
            #
            ax0.set_yscale('log')
            #
            plt.savefig(dir_path+os.sep+"tseries_overlap.png")
    if False: # Ruapehu unrest 2022
        sta='FWVZ'#'FWVZ'
        #look_back = .5 # days before eruption
        #look_front = 1.5 # days after eruption
        # station (volcano) and list of eruption times 
        import time as _time
        if sta == 'FWVZ':
            period = ['2022-01-01','2022-07-31 00:00:00']
        print('Station: '+sta+'\n')
        print('Inversion period. From '+period[0]+' to '+period[1]+'\n')
        #
        periods = _period_into_months(period[0],period[1])
        # loop over periods
        for i,p in enumerate(periods):
            newpath = r'.'+os.sep+'mcmc_inv_ruap_unrest_2022'+period[0].split('-')[0]+os.sep+''+str(p[0]).split(' ')[0] 
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            # directory to save file (inside of newpath, or by default in current directory)
            _dir = newpath 
            #
            st = _time.time()
            # # loop over tes to run inversion 
            ti=p[0]
            tf=p[1]
            if sta == 'FWVZ':
                fl= 'C:'+os.sep+'Users'+os.sep+'aar135'+os.sep+'codes_local_disk'+os.sep+'volc_forecast_tl'+os.sep+'volc_forecast_tl'+os.sep+'data'+os.sep+'FWVZ_tremor_data.csv'
                #fl= '..'+os.sep+'..'+os.sep+'data'+os.sep+'FWVZ_tremor_data.csv'
            # import data and select downsampling
            df = _load_tremor_data(ti,tf, downsample=1, fl = fl)
            print('Interting from '+str(p[0]).split(' ')[0]+' to '+str(p[1]).split(' ')[0])
            print('Points to invert: '+str(len(df)))
            # run mcmc
            run_mcmc_aux(df, _dir)
            # get the end time
            et = _time.time()
            # get the execution time
            elapsed_time = et - st
            if i == 0:
                print('Execution time:', elapsed_time, 'seconds\n')
            # plots
            plot_pars(_dir=_dir)
            #plot_results_mcmc(fl='./chains/chain30.dat') 
            #re_read_chains(fl_df='df_pars.csv')
            plot_pars_pers(_dir=_dir)
            # move files to directory 
            if _dir is None:
                os.rename('.'+os.sep+'chains', newpath+os.sep+'chains')
                os.rename('.'+os.sep+'df_pars.csv', newpath+os.sep+'df_pars.csv')
                os.rename('.'+os.sep+'df_pars_pers.csv', newpath+os.sep+'df_pars_pers.csv')
                os.rename('.'+os.sep+'pars_fit.png', newpath+os.sep+'pars_fit.png')
                os.rename('.'+os.sep+'pars_fit_pers.png', newpath+os.sep+'pars_fit_pers.png')
if __name__ == "__main__":
    main()