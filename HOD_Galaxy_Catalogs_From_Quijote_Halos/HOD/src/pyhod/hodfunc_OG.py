import numpy as np
from scipy.special import erf
from scipy.stats import bernoulli
from scipy.stats import poisson
from scipy.special import lambertw
from sklearn.neighbors import NearestNeighbors
from numba import jit, prange, set_num_threads
#testing:
import pyhod.read as readcats
import readgadget  #from pylians3
from scipy import special
   
def rnfw(n, con):
    ''' Sample from the NFW as a true PDF. Returns random samples ranges from 0–1 for "n" number of samples.
        
        Modified function from https://github.com/CullanHowlett/NFWdist, https://arxiv.org/pdf/1805.09550.pdf
        the cumulative probability, p, of finding a galaxy at radius q is p = M(q)/M(1). 
        Following the standard method of drawing from a PDF, we seek to generate
        random values for p ∈ [0, 1] and invert the CDF       
        
        input:
        ------
            n   : int
                number of samples
            con : float
                concentration parameter
        returns:
        --------
            p : array_like
                shape = (n,)
    '''
    pnfwunorm = lambda y: np.log(1.0 + y)-y/(1.0 + y) ##The dark matter mass enclosed within radius q
    p = np.random.uniform(low=0., high=1., size=n)
    p *=pnfwunorm(con)
    return (-(1.0/np.real(lambertw(-np.exp(-p-1))))-1)/con


def cent_sat(dat,logMmin, sigmalogM, logM0, alpha, logM1, params = None):
    ''' COMPUTE MEAN NUMBER OF GALAXIES
        The numbers of centrals and satellites for each halo are drawn from Bernoulli and Poisson distribution, respectively.

        input:
        ------
            dat   : array_like
                shape = (n,4); (mass, pos_x, pos_y, pos_y)
            params : dictionary
                keys = 'model' : Valid entries ('Zheng', 'exponential')
                refer to input_params() for more information.

            logMmin, sigmalogM, logM0, alpha, logM1 : float
            
        returns:
        --------
            N_cen : array_like
                shape = (n,)

            N_sat : array_like
                shape = (n,)
       '''
    np.seterr(invalid='ignore')
    print('\nComputing mean number central and satellites...')


    #The mean number of central galaxies and satellites:
    if params['model'] == 'Zheng':#refeq(2.2-2.3) arxiv:0703457
        N_mean_cen = 0.5*(1+erf((np.log10(dat[:,0]) - logMmin)/sigmalogM))
        N_mean_sat = N_mean_cen*np.power((dat[:,0] - 10**logM0)/10**logM1, alpha)

    elif params['model'] == 'exponential': #refeq(5-6) arxiv:0802.4288
        N_mean_cen = np.exp(-10**logMmin/dat[:,0])
        N_mean_sat = N_mean_cen*np.power(dat[:,0]/10**logM1,alpha)

    else:
        print("Parameter model not implemented.\n Valid entries are params['model'] = 'Zheng', 'exponential'\n")
        exit()

    N_mean_sat = np.nan_to_num(N_mean_sat, nan=0)
    
    print('\nDrawing central and satellites...')

    N_cen = bernoulli.rvs(N_mean_cen, random_state=params['seed'])
    N_sat = poisson.rvs(N_mean_sat, random_state=params['seed'])

    return np.array(N_cen), np.array(N_sat)


def galaxyPos_NFW(dat, N_cen, N_sat, params = None):
    ''' SAMPLE POSITION AND VELOCITES. 
        GALAXIES ARE DRAWN FROM NFW Profile AND VELOCTIES FROM A GAUSSIAN DISTRIBUTION
        WITH STD EQUAL TO THE HALO MEAN ORBITAL SPEED.

        input:
        ------
            dat   : array_like
                shape = (n,7); (mass, pos_x, pos_y, pos_y, vel_x, vel_y, vel_z)

            N_cen : array_like
                shape = (n,)

            N_sat : array_like
                shape = (n,)

            params : dictionary
                keys = 'om', 'z', 'ol', 'Deltac' , 'M_star' ,'box' : float
               refer to input_params() for more information.

        returns:
        --------
            gal_pos : array_like
            shape = (n,3)

            gal_vel : array_like
            shape = (n,3)

            gal_type : array_like
            shape = (n,)            
       '''
    
    print('\nAssigning position and velocities for satellites galaxies...\n')

    ##some parameters
    Ez  = np.sqrt(params['om']*(1.0+params['z'])**3+params['ol'])   #in km/s/(Mpc/h)
    rho_cri_z = 27.75e+10*Ez**2  ## critical density in units: [h^2*M_sun/Mpc^3].
    rho_vir   = params['om']*(params['Deltac']*rho_cri_z)  ## rho_vir=omega_m*rho_200.
    
    c0, beta, Ms0 = 9, -0.13, params['M_star'] ## Note: c0 and beta are hardcoded. Check eqref(1) arxiv:0703457 for more info. 

    ## select halos that host central galaxies
    mask       = np.bool_(N_cen)
    cen_pos    = dat[mask,1:4]   ##[Mpc/h]
    cen_vel    = dat[mask,4:7]   ##[km/s]
    mass_vir   = dat[mask,0]     ##[Msun/h]
    masked_sat = N_sat[mask]

    ## mask over central galaxies that host satellites
    mask = masked_sat>0
    masked_sat = masked_sat[mask].astype(int)
    totsat = np.sum(masked_sat).astype(int)

    ## Compute the NFW profile
    mass_vir = np.repeat(mass_vir[mask], masked_sat)
    r_vir    = np.cbrt(3/4/np.pi/rho_vir*mass_vir)   ##[Mpc/h]
    conp     = c0/(1+params['z'])*(mass_vir/Ms0)**beta ##concentration parameter, where conp=Rvir/Rs. Check eqref(1) arxiv:0703457
    
# radial satellites positions:
    r = rnfw(totsat,conp)
    r_sat = r*r_vir            ##[Mpc/h] unitary radial position
    costheta = np.random.uniform(low=-1., high=1., size = totsat)
    sintheta = np.sqrt(1-costheta**2)
    phi      = np.random.uniform(low=0., high=1., size = totsat)*2*np.pi
    
    ## satellites positions:
    sat_pos = np.repeat(cen_pos[mask],masked_sat,axis=0)
    sat_pos[:,0] += r_sat*np.cos(phi)*sintheta    ##[Mpc/h]
    sat_pos[:,1] += r_sat*np.sin(phi)*sintheta    ##[Mpc/h]
    sat_pos[:,2] += r_sat*costheta                ##[Mpc/h]
    sat_pos  = (sat_pos + params['box']) % params['box']  ##[Mpc/h] boundary conditions to box

    # Drawn from a Gaussian distribution with std = mean orbital speed
    ## satellites velocities:
    sat_vel = np.repeat(cen_vel[mask],masked_sat,axis=0)
    sigmav  = np.sqrt(4.3009e-9*mass_vir/r_vir)  # virial velocity in units [km/s], with G = 4.3009e-9
    sat_vel += np.random.normal(loc=0., scale=sigmav/np.sqrt(2), size=(3,totsat)).T
    

    ## wrap central and satellites
    gal_pos  = np.vstack([cen_pos, sat_pos])
    gal_vel  = np.vstack([cen_vel, sat_vel])

    totcen   = len(cen_pos)
    gal_type = np.zeros(totcen + totsat)
    gal_type[:totcen] = 1

    halo_indx = np.arange(totcen)
    sat_indx = np.repeat(halo_indx[mask], masked_sat)
    halo_indx = np.concatenate([halo_indx,sat_indx])

    print('\nTotal number of galaxies:', totcen+totsat)
    print('Total number of central:', totcen)
    print('Total number of satellites: %i\n'%totsat)

    return gal_pos, gal_vel, gal_type, halo_indx


def galaxyPos_NFWv(dat, N_cen, N_sat, params = None):
    '''SAMPLE POSITION AND VELOCITES. 
       GALAXIES AND VELOCTIES ARE DRAWN FROM NFW Profile

        input:
        ------
            dat   : array_like
                shape = (n,7); (mass, pos_x, pos_y, pos_y, vel_x, vel_y, vel_z)

            N_cen : array_like
                shape = (n,)

            N_sat : array_like
                shape = (n,)

            params : dictionary
                keys = 'om', 'z', 'ol', 'Deltac' , 'M_star' ,'box' : float
               refer to input_params() for more information.

        returns:
        --------
            gal_pos : array_like
            shape = (n,3)

            gal_vel : array_like
            shape = (n,3)

            gal_type : array_like
            shape = (n,)            
       '''
    
    print('\nAssigning position and velocities for satellites galaxies...\n')

    ##some parameters
    Ez  = np.sqrt(params['om']*(1.0+params['z'])**3+params['ol'])   #in km/s/(Mpc/h)
    rho_cri_z = 27.75e+10*Ez**2  ## critical density in units: [h^2*M_sun/Mpc^3].
    rho_vir   = params['om']*(params['Deltac']*rho_cri_z)  ## rho_vir=omega_m*rho_200.

    c0,beta, Ms0 = 9, -0.13, params['M_star']

    ## select halos that host central galaxies
    mask       = np.bool_(N_cen)
    cen_pos    = dat[mask,1:4]   #[Mpc/h]
    cen_vel    = dat[mask,4:7]    #[km/s]
    mass_vir   = dat[mask,0]     #[Msun/h]
    masked_sat = N_sat[mask]

    ## mask over central galaxies that host satellites
    mask = masked_sat>0
    masked_sat = masked_sat[mask].astype(int)
    totsat = np.sum(masked_sat).astype(int)

    #Compute the NFW profile
    mass_vir = np.repeat(mass_vir[mask], masked_sat)
    r_vir    = np.cbrt(3/4/np.pi/rho_vir*mass_vir)   ##[Mpc/h]
    conp     = c0/(1+params['z'])*(mass_vir/Ms0)**beta ##concentration parameter, where conp=Rvir/Rs. Check eqref(1) arxiv:0703457
    
    r = rnfw(totsat,conp)*r_vir ##[Mpc/h]
    costheta = np.random.uniform(low=-1., high=1., size = totsat)
    sintheta = np.sqrt(1-costheta**2)
    phi      = np.random.uniform(low=0., high=1., size = totsat)*2*np.pi
    
    ## satellites positions:
    sat_pos = np.repeat(cen_pos[mask],masked_sat,axis=0)
    sat_pos[:,0] += r*np.cos(phi)*sintheta    ##[Mpc/h]
    sat_pos[:,1] += r*np.sin(phi)*sintheta    ##[Mpc/h]
    sat_pos[:,2] += r*costheta                ##[Mpc/h]
    sat_pos  = (sat_pos + params['box']) % params['box']  ##[Mpc/h]
    # NFW profile circular velocity at location r:

    # NFW profile circular velocity at location r:
    r_s= r_vir/conp
    s = 0.5*(conp + conp*conp - (1+conp)*np.log(1+conp))/((1+conp)*np.log(1+conp)-conp)**2
    sigmav = np.sqrt(4.3009e-9*mass_vir/r_s*s)
    sat_vel = np.repeat(cen_vel[mask],masked_sat,axis=0) 
    sat_vel += np.random.normal(loc=0, scale=sigmav/np.sqrt(3), size=(3,totsat)).T


    ## wrap central and satellites
    gal_pos  = np.vstack([cen_pos, sat_pos])
    gal_vel  = np.vstack([cen_vel, sat_vel])

    totcen   = len(cen_pos)
    gal_type = np.zeros(totcen + totsat)
    gal_type[:totcen] = 1 

    halo_indx = np.arange(totcen)
    sat_indx = np.repeat(halo_indx[mask], masked_sat)
    halo_indx = np.concatenate([halo_indx,sat_indx])

    print('\nTotal number of galaxies:', totcen+totsat)
    print('Total number of central:', totcen)
    print('Total number of satellites: %i\n'%totsat)

    return gal_pos, gal_vel, gal_type, halo_indx


def galaxyPos_SNAPknn(dat, N_cen, N_sat, params = None):
    '''SAMPLE POSITION AND VELOCITES FROM A SNAPSHOT OF DM PARTICLES
       USING A KNN ALGORITHM (VERY SLOW AND NOT OPTIMIZED) AND GALAXIES ARE DRAWN FROM A UNIFORM DISTRIBUTION.

        input:
        ------
            dat   : array_like
                shape = (n,4); (mass, pos_x, pos_y, pos_y, vel_x, vel_y, vel_z)

            N_cen : array_like
                shape = (n,)

            N_sat : array_like
                shape = (n,)

            params : dictionary
                keys = 'om', 'z', 'ol', 'Deltac' , 'M_star' : float
                        'ifile_snapshot' : str
               refer to input_params() for more information.

        returns:
        --------
            gal_pos : array_like
            shape = (n,3)

            gal_vel : array_like
            shape = (n,3)

            gal_type : array_like
            shape = (n,)      
       '''
    
    print('\nAssigning position and velocities for satellites galaxies...\n')

    ##some parameters
    Ez  = np.sqrt(params['om']*(1.0+params['z'])**3+params['ol'])   #in km/s/(Mpc/h)
    rho_cri= 27.75e+10*Ez**2  ## critical density in units: [h^2*M_sun/Mpc^3]
    rho_vir = params['om']*(params['Deltac']*rho_cri)*(1.0+params['z'])**3  # ; rho_vir=omega_m*rho_200.

    ## select halos that host central galaxies
    mask       = np.bool_(N_cen)
    cen_pos    = dat[mask,1:4]   #[Mpc/h]
    cen_vel    = dat[mask,4:7]   #[km/s]
    mass_vir   = dat[mask,0]     #[Msun/h]
    group_len  = dat[mask,7]     #Number of DM particles in each halo
    masked_sat = N_sat[mask]
    
    ## mask over central galaxies that host satellites
    mask = masked_sat>0
    masked_sat = masked_sat[mask].astype(int)
    totsat = np.sum(masked_sat).astype(int)

    r_vir   = np.cbrt(3/4/np.pi/rho_vir*mass_vir[mask])   ##[Mpc/h]
    masked_cen_pos = cen_pos[mask]   #[Mpc/h]
    group_len = group_len[mask].astype(int)
    
    print('drawing galaxies from snapshot: ', params['ifile_snapshot'])
    #vec_DM = readcatalogs.readcats(params['ifile_snapshot'])   ## units: mass: [Msun/h], pos:[Mpc/h], vel:[km/s]
    ptype   = 1  #CDM
    pos_DM  = readgadget.read_block(params['ifile_snapshot'],"POS ",[ptype])/1e3 #Mpc/h
    vel_DM  = readgadget.read_block(params['ifile_snapshot'],"VEL ",[ptype]) #km/s
    mass_DM = readgadget.read_block(params['ifile_snapshot'],"MASS",[ptype])*1e10 #mass of the particles Msun/h

    vec_DM = np.array([mass_DM[:], pos_DM[:,0], pos_DM[:,1], pos_DM[:,2], vel_DM[:,0], vel_DM[:,1], vel_DM[:,2]]).T
    sat_pos = np.zeros((totsat,3),dtype=np.int32)
    sat_vel = np.zeros((totsat,3),dtype=np.int32)

    temp = 0 #number to iterate sat_pos, sat_vel
    for i in range(len(masked_cen_pos)):
        
        pos_h = masked_cen_pos[i] #central position of the ith-halo hosting satelites
        gal_len = masked_sat[i]   #Number of satelites in the Halo
        grouplen = group_len[i]   #Number of DM particles inside the Halo

        #slice snapshot around the halo center with width 5*r_vir
        mask =(vec_DM[:,1]>pos_h[0]-5*r_vir[i])&(vec_DM[:,1]<pos_h[0]+5*r_vir[i])
        vecDM_slice = vec_DM[mask,:]

        mask= (vecDM_slice[:,2]>pos_h[1]-5*r_vir[i])&(vecDM_slice[:,2]<pos_h[1]+5*r_vir[i])
        vecDM_slice = vecDM_slice[mask,:]

        mask= (vecDM_slice[:,3]>pos_h[2]-5*r_vir[i])&(vecDM_slice[:,3]<pos_h[2]+5*r_vir[i])
        vecDM_slice = vecDM_slice[mask,:]

        #kNN = NearestNeighbors(n_neighbors=grouplen,n_jobs=32)
        kNN = NearestNeighbors(n_neighbors=grouplen)
        kNN.fit(vecDM_slice[:,1:4])
        distances, indices = kNN.kneighbors([pos_h])

        ind = np.random.choice(np.shape(indices)[1], gal_len, replace=False)
        
        ind = indices[0,ind]

        sat_pos[temp:temp+gal_len,:] = vecDM_slice[ind,1:4]
        sat_vel[temp:temp+gal_len,:] = vecDM_slice[ind,4:7]

        temp += gal_len

    ## wrap central and satellites
    gal_pos  = np.vstack([cen_pos, sat_pos])
    gal_vel  = np.vstack([cen_vel, sat_vel])

    totcen   = len(cen_pos)
    gal_type = np.zeros(totcen + totsat)
    gal_type[:totcen] = 1 

    halo_indx = np.arange(totcen)
    sat_indx = np.repeat(halo_indx[mask], masked_sat)
    halo_indx = np.concatenate([halo_indx,sat_indx])


    print('\nTotal number of galaxies:', totcen+totsat)
    print('Total number of central:', totcen)
    print('Total number of satellites: %i\n'%totsat)

    return gal_pos, gal_vel, gal_type, halo_indx


@jit(nopython=True,fastmath = True,parallel = True)
def drawnFromSnap(totsat,sat_len, sat_indx, r_vir, cen_pos, group_len, vec_DM):
    ''' GIVEN A SNAPSHOT OF DM PARTICLES DRAWN SATELLITES POSTION AND VELOCITIES
        FROM SUBBOX OF THE SNAPSHOT CENTERED AT THE HALO_CENTER AND RADIUS VIRAL_RAIDUS OF THE HOSTING HALO.
        GALAXIES ARE DRAWN FROM A UNIFORM DISTRIBUTION
        RUTINE IMPLEMENTED IN galaxyPos_SNAP().
       '''
    set_num_threads(int(4))
    N_cen = len(cen_pos)
    pos_temp = np.empty((totsat, 3))
    vel_temp = np.empty((totsat, 3))  
    pos_temp[:] = 0
    vel_temp[:] = 0
    print('drawing galaxies from snapshot',N_cen)
    for i in prange(N_cen):
        pos_h = cen_pos[i] #central position of the ith-halo hosting satelites
        gal_len = sat_len[i]   #Number of satelites in the Halo
        grouplen = group_len[i]   #Number of DM particles inside the Halo

        #slice snapshot around the halo center with width r_vir
        mask =(vec_DM[:,1]>pos_h[0]-r_vir[i])&(vec_DM[:,1]<pos_h[0]+r_vir[i])
        vecDM_slice = vec_DM[mask,:]
     
        mask= (vecDM_slice[:,2]>pos_h[1]-r_vir[i])&(vecDM_slice[:,2]<pos_h[1]+r_vir[i])
        vecDM_slice = vecDM_slice[mask,:]
        
        mask= (vecDM_slice[:,3]>pos_h[2]-r_vir[i])&(vecDM_slice[:,3]<pos_h[2]+r_vir[i])
        vecDM_slice = vecDM_slice[mask,:]
        
        r_slice = np.sqrt((vecDM_slice[:,1]-pos_h[0])**2+(vecDM_slice[:,2]-pos_h[1])**2 + (vecDM_slice[:,3]-pos_h[2])**2)
        mask = r_slice<r_vir[i]
        vecDM_slice = vecDM_slice[mask,:]

        ind = np.random.choice(np.shape(vecDM_slice)[0], gal_len, replace=False)

        fill_from = sat_indx[i]
        fill_to = sat_indx[i]+gal_len

        pos_temp[fill_from:fill_to] = vecDM_slice[ind,1:4]
        vel_temp[fill_from:fill_to] = vecDM_slice[ind,4:7]

    return pos_temp, vel_temp


def galaxyPos_SNAP(dat, N_cen, N_sat, params = None):
    ''' SAMPLE POSITION AND VELOCITES FROM A SNAPSHOT OF DM PARTICLES
        GALAXIES ARE DRAWN FROM A UNIFORM DISTRIBUTION FROM SUBBOX OF THE SNAPSHOT CENTERED 
        AT THE HALO_CENTER AND RADIUS VIRAL_RAIDUS OF THE HOSTING HALO.      

        input:
        ------
            dat   : array_like
                shape = (n,7); (mass, pos_x, pos_y, pos_y)

            N_cen : array_like
                shape = (n,)

            N_sat : array_like
                shape = (n,)

            params : dictionary
                keys = 'om', 'z', 'ol', 'Deltac' , 'M_star' : float
                        'ifile_snapshot' : str
               refer to input_params() for more information.

        returns:
        --------
            gal_pos : array_like
            shape = (n,3)

            gal_vel : array_like
            shape = (n,3)

            gal_type : array_like
            shape = (n,)
    '''

    print('\nAssigning position and velocities for satellites galaxies...\n')

    ##some parameters
    #Ez  = np.sqrt(params['om']*(1.0+params['z'])**3+params['ol'])   #in km/s/(Mpc/h)
    rho_cri= 27.75e+10#*Ez**2  ## critical density in units: [h^2*M_sun/Mpc^3]
    rho_vir = params['om']*(params['Deltac']*rho_cri)*(1.0+params['z'])**3  # ; rho_vir=omega_m*rho_200.

    ## select halos that host central galaxies
    mask       = np.bool_(N_cen)
    cen_pos    = dat[mask,1:4]   #[Mpc/h]
    cen_vel    = dat[mask,4:7]   #[km/s]
    mass_vir   = dat[mask,0]     #[Msun/h]
    group_len  = dat[mask,7]     #Number of DM particles in each halo
    masked_sat = N_sat[mask]
    
    ## mask over central galaxies that host satellites
    mask = masked_sat>0
    masked_sat = masked_sat[mask].astype(int)
    totsat = np.sum(masked_sat).astype(int)

    r_vir   = np.cbrt(3/4/np.pi/rho_vir*mass_vir[mask])   ##[Mpc/h]
    masked_cen_pos = cen_pos[mask]   #[Mpc/h]
    group_len = group_len[mask].astype(int)
    
    print('loading snapshot: ', params['ifile_snapshot'])
    #vec_DM = readcatalogs.readcats(params['ifile_snapshot'])   ## units: mass: [Msun/h], pos:[Mpc/h], vel:[km/s]
    # read the positions and veloccities of the particles
    ptype   = 1  #CDM
    pos_DM  = readgadget.read_block(params['ifile_snapshot'],"POS ",[ptype])/1e3 #Mpc/h
    vel_DM  = readgadget.read_block(params['ifile_snapshot'],"VEL ",[ptype]) #km/s
    mass_DM = readgadget.read_block(params['ifile_snapshot'],"MASS",[ptype])*1e10 #mass of the particles Msun/h

    vec_DM = np.array([mass_DM[:], pos_DM[:,0], pos_DM[:,1], pos_DM[:,2], vel_DM[:,0], vel_DM[:,1], vel_DM[:,2]]).T
    ## These are the indx to fill sat_pos and set_vel in drawnFromSnap()
    sat_indx=np.zeros(len(masked_sat)+1)
    sat_indx[1:]=np.cumsum(masked_sat)

    sat_pos, sat_vel = drawnFromSnap(totsat,masked_sat,sat_indx,r_vir,masked_cen_pos,group_len,vec_DM)
     
    ## wrap central and satellites
    gal_pos  = np.vstack([cen_pos, sat_pos])
    gal_vel  = np.vstack([cen_vel, sat_vel])

    totcen   = len(cen_pos)
    gal_type = np.zeros(totcen + totsat)
    gal_type[:totcen] = 1 

    halo_indx = np.arange(totcen)
    sat_indx = np.repeat(halo_indx[mask], masked_sat)
    halo_indx = np.concatenate([halo_indx,sat_indx])


    print('\nTotal number of galaxies:', totcen+totsat)
    print('Total number of central:', totcen)
    print('Total number of satellites: %i\n'%totsat)

    return gal_pos, gal_vel, gal_type, halo_indx


def sample_catalog(halos, centrals, satellites, params=None):
    ''' SAMPLE POSITION AND VELOCITES FROM A SNAPSHOT OF DM PARTICLES  

        input:
        ------
            halos   : array_like
                shape = (n,4); (mass, pos_x, pos_y, pos_y, vel_x, vel_y, vel_z)

            centrals : array_like
                shape = (n,3)

            satellites : array_like
                shape = (n,3)

            params : dictionary
                keys = 'om', 'z', 'ol', 'Deltac' , 'M_star' : float
                        'ifile_snapshot' : str
               refer to input_params() for more information.

        returns:
        --------
            gal_pos : array_like
            shape = (n,3)

            gal_vel : array_like
            shape = (n,3)

            gal_type : array_like ; 1 for central galaxy, 0 = for satallite galaxy.
            shape = (n,) 
    '''

    np.random.seed(params['seed'])

    if params['method'] == 'NFW':
        print('Doing NFW')
        galaxy_pos, galaxy_vel, galaxy_type, halo_indx = galaxyPos_NFW(halos, centrals, satellites, params)
        
    if params['method'] == 'NFWv':
        print('Doing NFWv')
        galaxy_pos, galaxy_vel, galaxy_type, halo_indx = galaxyPos_NFWv(halos, centrals, satellites, params)

    if params['method'] == 'SNAP':
        print('Doing SNAP')
        galaxy_pos, galaxy_vel, galaxy_type, halo_indx = galaxyPos_SNAP(halos, centrals, satellites, params)

    if params['method']== 'SNAPknn':
        print('Doing SNAPknn')
        galaxy_pos, galaxy_vel, galaxy_type, halo_indx = galaxyPos_SNAPknn(halos, centrals, satellites,params)

    return  galaxy_pos, galaxy_vel, galaxy_type, halo_indx