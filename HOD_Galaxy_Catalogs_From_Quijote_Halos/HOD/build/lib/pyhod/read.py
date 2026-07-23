import numpy as np
#sys.path.append('/Users/matteobiagetti/GitProjects/py-power/src/')
import pyhod.DataReading.readfof as readfof
import pyhod.DataReading.readgadget as readgadget # Issue with Quijote data: check https://quijote-simulations.readthedocs.io/en/latest/snapshots.html for Warning
#import readgadget  #load from Pylians3

def readcats(params = None):
    ''' Helper function to load data.

        input:
        ------
            params : dictionary
                params['ifile'] : str
                    Input catalog filename
                params['cat']  : str
                    type of catalog. Valid entries ('EoS', 'Quijote_Halo')
                params['z']  : float
                    redshift of the catalog
            refer to input_params() for more information.

        returns:
        --------
            vec : array_like
            shape = (mass, pos_x, pos_y, pos_y, vel_x, vel_y, vel_y)    
    '''
    print('\n\nLoading halos from\n    --->FILE:%s'%params['ifile'])
    
    if params['cat'] == 'Eos':
        mycat = np.load(params['ifile'])

        vec   = np.array([mycat['mass'][:], mycat['pos'][0,:], mycat['pos'][1,:], mycat['pos'][2,:], mycat['vel'][0,:], mycat['vel'][1,:], mycat['vel'][2,:]]).T

    if params['cat'] == 'Quijote_Halo':
        # determine the redshift, z, of the catalogue
        #z_dict = {4:0.0, 3:0.5, 2:1.0, 1:2.0, 0:3.0}
        #redshift  = z_dict[snapnum]
        z_dict  = {0.0:4, 0.5:3, 1.0:2, 2.0:1, 3.0:0} #{redshift:snapnum}
        snapnum = z_dict[params['z']]
        # read the halo catalogue
        FoF = readfof.FoF_catalog(params['ifile'], snapnum, long_ids=False, swap=False, SFR=False, read_IDs=False)
        # get properties of the halos
        mass  = FoF.GroupMass*1e10              #Halo masses in Msun/h
        pos_h = FoF.GroupPos/1e3                #Halo positions in Mpc/h
        vel_h = FoF.GroupVel*(1.0+params['z'])  #Halo peculiar velocities in km/s
        halos_len = FoF.GroupLen                #Number of CDM particles in the halo
        
        vec = np.array([mass[:], pos_h[:,0], pos_h[:,1], pos_h[:,2], vel_h[:,0], vel_h[:,1], vel_h[:,2], halos_len[:]]).T

    if params['cat'] == 'Quijote_Snap':
        # read the positions and veloccities of the particles
        ptype   = 1  #CDM
        pos_DM  = readgadget.read_block(params['ifile'],"POS ",[ptype])/1e3 #Mpc/h
        vel_DM  = readgadget.read_block(params['ifile'],"VEL ",[ptype]) #km/s
        mass_DM = readgadget.read_block(params['ifile'],"MASS",[ptype])*1e10 #mass of the particles Msun/h

        vec = np.array([mass_DM[:], pos_DM[:,0], pos_DM[:,1], pos_DM[:,2], vel_DM[:,0], vel_DM[:,1], vel_DM[:,2]]).T

    return vec
