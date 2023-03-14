import numpy as np
from time import time
from scipy.interpolate import RegularGridInterpolator
from obspy.signal.filter import bandpass 
from datetime import datetime, timedelta, date
import random
import os
#
# define constant
pi=np.pi
sqrt=np.sqrt
_MONTH = timedelta(days=365.25/12)
_DAY = timedelta(days=1.)
_HOUR = timedelta(days=1./24)
_MIN = timedelta(days=1./24/60)
BANDS=['rsam','mf','hf']#=['vlf','lf','rsam','mf','hf']
RATIOS=['rmar','dsar']
FILTERED_DATA=True # EQ filtered data. Add 'F' when loading
FBANDS=[[2, 5], [4.5, 8], [8,16]] #FBANDS=[[0.01,0.1],[0.1,2],[2, 5], [4.5, 8], [8,16]]
#
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
        R=200;                                                          #conduit radius
        L=50;                                                           #thickness of the cap
        distance=2000;                                                  #source-receiver distance
        max_freq=50;                                                    #maximum frequency to be reached in the simulation
    # WIZ
    if True:
        tau=30;#600                                                     #seconds of simulation
        R=100;                                                          #conduit radius
        L=50;                                                           #thickness of the cap
        distance=1000;                                                  #source-receiver distance
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
    Qf=40;                                                                      #quality factor. 
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
    if False:#Pressure:
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
        _datas.append(_data)
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
def create_eos():
    # check directory structure
    newpath = r'.'+os.sep+'lookup_tables' 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    # list of parameter dimensions
    D=3   # number of dimensions
    nD=48 # number of parameter divisions in each dimension
    print('runs '+str(nD**D)+' ET: '+str(nD**D*.2/3600)+' hours')
    #ps=[np.linspace(0,1,nD) for i in range(D)]    # list of parameter vectors (simulation inputs)
    # set the initial location of the walkers
    _D=np.logspace(-3, 1, num=nD)
    _Q=np.linspace(1,100,nD)
    _N=np.logspace(0, 4, num=nD)
    ps = [_D,_Q,_N]
    #
    #eos_rsam=np.random.rand(*[nD for i in range(D)])*0     # fill interpolating array with random values (simulation outputs)
    eos_rsam=np.random.rand(*[nD for i in range(D)])*0 
    eos_mf=np.random.rand(*[nD for i in range(D)])*0 
    eos_hf=np.random.rand(*[nD for i in range(D)])*0 
    eos_dp=np.random.rand(*[nD for i in range(D)])*0 
    if RATIOS:
        eos_rmar=np.random.rand(*[nD for i in range(D)])*0 
        eos_dsar=np.random.rand(*[nD for i in range(D)])*0 

    # fill eos
    t1=time()
    for i,d in enumerate(_D):
        for j,q in enumerate(_Q):
            for k,n in enumerate(_N):
                DP, DP2, U_z, V_z=fwd_cav_Ps_Uz_Vz(D=d,Q=q,N=int(n))
                datas = Vz_filters(V_z)
                if RATIOS:
                    _datas= calc_ratios(U_z)
                #
                #eos[i,j,k]=[np.mean(np.abs(d)) for d in datas]
                eos_rsam[i,j,k]=np.mean(np.abs(datas[0]))
                eos_mf[i,j,k]=np.mean(np.abs(datas[1]))
                eos_hf[i,j,k]=np.mean(np.abs(datas[2]))
                eos_dp[i,j,k]=np.max(np.real(DP2))
                if RATIOS:
                    eos_rmar[i,j,k]=np.mean(np.abs(_datas[0]))
                    eos_dsar[i,j,k]=np.mean(np.abs(_datas[1]))

    t2=time()
    print((t2-t1)/(i*j*k),'seconds per run')
    #
    if False: # test speed of rgi
        rgi=RegularGridInterpolator(ps, eos_rsam)       # create look-up table
        # time look-up table
        N=100   # number of look-ups
        t1=time()
        for i in range(N):
            p=np.array([_D[0],_Q[0],_N[0]])    # random parameter vector
            rgi(p)                 # look up simulation result
        t2=time()
        print((t2-t1)/N,'seconds per run')

    # save and load 
    np.save(newpath+os.sep+'eos_rsam.npy', eos_rsam)
    np.save(newpath+os.sep+'eos_mf.npy', eos_mf)
    np.save(newpath+os.sep+'eos_hf.npy', eos_hf)
    np.save(newpath+os.sep+'eos_dp.npy', eos_dp)
    np.save(newpath+os.sep+'eos_rmar.npy', eos_rmar)
    np.save(newpath+os.sep+'eos_dsar.npy', eos_dsar)
    #
def load_eos(fl = 'eos_rsam.npy'):
    '''
    '''
    # list of parameter dimensions
    D=3   # number of dimensions
    nD=12#10 # number of parameter divisions in each dimension
    _D=np.logspace(-3, 1, num=nD)
    _Q=np.linspace(1,100,nD)
    _N=np.logspace(0, 4, num=nD)
    ps = [_D,_Q,_N]
    #
    eos_rsam=np.load(fl)
    #
    if True: # test speed of rgi
        rgi=RegularGridInterpolator(ps, eos_rsam)       # create look-up table
        # time look-up table
        N=100   # number of look-ups
        t1=time()
        for i in range(N):
            p=np.array([_D[0],_Q[0],_N[0]])    # random parameter vector
            rgi(p)                 # look up simulation result
        t2=time()
        print((t2-t1)/N,'seconds per run')

if __name__=="__main__":
    create_eos()
    #load_eos(fl = 'lookup_tables'+os.sep+'eos_rsam.npy')
