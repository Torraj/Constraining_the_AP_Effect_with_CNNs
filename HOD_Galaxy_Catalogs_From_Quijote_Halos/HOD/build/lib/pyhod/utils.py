import numpy as np

def displaceRSD(gal_pos, gal_vel, params = None):
    ''' DISPLACE GALAXIES TO REDSHIFT SPACE DISTORTION

        input:
        ------
            gal_pos : array_like
            shape = (n,3)

            gal_vel : array_like
            shape = (n,3)

            params : dictionary
                keys = 'om', 'z', 'ol', 'iRSD' , 'zerr' : float
               refer to input_params() for more information.

        returns:
        --------
            gal_pos : array_like
            shape = (n,3)
    '''

    zerrdic = {1:'type: spectroscopic.',2:'type: photometric.'}
    Hubble = 100.0*np.sqrt(params['om']*(1.0+params['z'])**3+params['ol'])   #in km/s/(Mpc/h)

    params['iRSD'] 
    params['zerr'] 
    if (params['iRSD']  < 4):
        print('Plane Parallel along %i...'%params['iRSD'])
        if params['zerr'] <2:
            print('Displacing galaxies to RSD%i'%params['iRSD'])
            gal_pos[:,params['iRSD']-1] += (1.+params['z'])/Hubble*gal_vel[:,params['iRSD']-1]
        if params['zerr'] >0:
            print('Drawn redshift error %s'%zerrdic[params['zerr'] ])
            sigmav = 60 if params['zerr'] ==1 else 3000 ################ THIS IS HARDCODED AT THE MOMMENT (USE MEAN VELOCITY DISPERSION OF THE CATALOG)
            gal_pos[:,params['iRSD']-1] += np.random.normal(loc=0, scale=(1+params['z'])/Hubble*sigmav,size=len(gal_pos[:,params['iRSD']-1]))
    elif params['iRSD']==5:
        print('Wide angle observer')
        r = np.sqrt(gal_pos[:,0]**2 + gal_pos[:,1]**2 + gal_pos[:,2]**2)
        if params['zerr']<2:
            print('Displacing galaxies to RSD%i'%params['iRSD'])
            gal_pos[:,0] += (1.+params['z'])/Hubble*gal_vel[:,0]*gal_pos[:,0]/r
            gal_pos[:,1] += (1.+params['z'])/Hubble*gal_vel[:,1]*gal_pos[:,1]/r
            gal_pos[:,2] += (1.+pparams['z'])/Hubble*gal_vel[:,2]*gal_pos[:,2]/r
        if params['zerr']>0:
            print('Drawn redshift error %s'%zerrdic[params['zerr']])
            sigma = 60 if params['zerr']==1 else 3000 ################ THIS IS HARDCODED AT THE MOMMENT (USE MEAN VELOCITY DISPERSION OF THE CATALOG)
            sigmav = np.random.normal(loc=0, scale=(1+params['z'])/Hubble*sigma,size=len(gal_pos[:,0]))
            gal_pos[:,0] += sigmav*gal_pos[:,0]/r
            gal_pos[:,1] += sigmav*gal_pos[:,1]/r
            gal_pos[:,2] += sigmav*gal_pos[:,2]/r
    gal_pos = (gal_pos + params['box']) % params['box']

    return gal_pos

def top_hat(k,R=8):
    '''
        Top hat window function in Fourier space
        
        input:
        ------
            avgk : array_like
            wavelengt in units of [h/Mpc]
            shape = (n,)
            R : float
                in units: [Mpc/h]

    '''
    return 3/(k*R)**3*(np.sin(k*R)-(k*R)*np.cos(k*R))

def M_star(avgk, avgP, R):
    '''
        TYPICAL COLLAPSING MASS PARAMETER

        input:
        ------
            avgk : array_like
                wavelengt in units of [h/Mpc]
                shape = (n,)

            avgP : array_like
                powerspectrum in units of [h/Mpc]^3
                shape = (n,)

            R : float
                in units: [Mpc/h]

        returns:
        --------
            M_star : float
                in units [M_sun/h]
    '''
    from scipy import integrate
    
    sigma_R = integrate.simpson(y=avgk**2*avgP*top_hat(avgk,R)**2,x=avgk)*0.5/np.pi**2
    print('sigma_R = ',np.sqrt(sigma_R))

    #the dimensionless “peak height” mass parameter nu(M)
    print('nu(R) = ',1.687/np.sqrt(sigma_R))

    ## the characteristic clustering mass, M⋆(z=0), defined by nu(M⋆) = 1.
    rho_cri = 27.75e10  #[h^2*M_sun/Mpc^3].
    M_star = rho_cri*4*np.pi*R**3/3  ##in units [M_sun/h]
    print('M_* = %e'%M_star)

    return M_star
