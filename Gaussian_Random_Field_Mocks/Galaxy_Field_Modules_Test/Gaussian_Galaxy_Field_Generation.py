import scipy.fftpack
from scipy.interpolate import CubicSpline
import numpy as np
import camb
from camb import model, initialpower

h = None
field_size = None
max_stretch = None

def initialize(h_dum, field_size_dum, max_stretch_dum):
    global h, field_size, max_stretch
    h = h_dum
    field_size = field_size_dum
    max_stretch = max_stretch_dum
    print("ggfg initialized!")


def fftind(size):
    """ Returns a numpy array of shifted Fourier coordinates k_x k_y k_z.
        
        Input args:
            size (integer): The size of the coordinate array to create
        Returns:
            k_ind, numpy array of shape (3, size, size, size) with:
                k_ind[0,:,:,:]:  k_x components
                k_ind[1,:,:,:]:  k_y components
                k_ind[2,:,:,:]:  k_z components
            
        """
    #Below, I have added an extra ':size' to the make the output 3D
    k_ind = np.mgrid[:size, :size, :size] - int( (size + 1)/2 )
    k_ind = scipy.fftpack.fftshift(k_ind) / (1000*h) * 2*np.pi
    return( k_ind )

def camb_matter_power_spectrum():
    """ Returns a realistic matter power spectrum (from camb package) evaluated at discrete points in k-space
        
        Input args:
            h (float): Dimensionless Hubble constant
            field_size (integer): The size of the cube output Gaussian/galaxy fields
        Returns:
            pk_nonlin[0], numpy array of shape (npoints): Matter power spectrum at redshift z=0 evaluated at
                                                          npoints within a specified range of k-values 
            kh_nonlin, numpy array of shape (npoints): Array of k-values over which to evaluate the power spectrum             
    """
    # Set cosmological parameters
    pars = camb.set_params(H0=67.5, ombh2=0.022, omch2=0.122, ns=0.965)
    
    # Get matter power spectra at redshift 0
    pars.set_matter_power(redshifts=[0.0], kmax=2.0)
    
    # Non-Linear spectra (Halofit)
    results = camb.get_results(pars)
    pars.NonLinear = model.NonLinear_both
    results.calc_power_spectra(pars)
    
    # In order to get a reasonable number of galaxies (~10^6), I need my minimum k to be 0.001
    # Maximum k should then be the number of voxels along one side of the field times the minimum
    # k. My fields won't be able to resolve the effects of larger k.
    
    mink = 2*np.pi*0.001 / (2*h) # Divided by 2 to account for maximum squashing
    kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh= mink, maxkh= field_size * mink * 4, npoints= 10000)
    # Multiplied by 4 to account for previous division by 2 and the maximum stretching
    return pk_nonlin[0], kh_nonlin

def stretched_power():
    stretch = np.random.uniform(1 / max_stretch, max_stretch)
    
    # For each position in k-space, I want to calculate the separation from the origin
    k_idx = fftind(field_size)
    
    # We stretch physical space along one or more of the axes by a factor of the stretch
    # Stretch factor multiplied by k_idx[0] stretches the images horizontally
    # Stretch factor multiplied by k_idx[2] stretches the images vertically
    k_sep = np.sqrt((k_idx[0] * stretch)**2 + (k_idx[1])**2 + (k_idx[2])**2)

    pk, kh = camb_matter_power_spectrum()
    
    # Because we have not defined the power spectrum for every k we want to evaluate, we interpolate
    power_interpolator = CubicSpline(kh, pk, extrapolate = False)
    amplitude = power_interpolator(k_sep)
    amplitude[0][0][0] = 0
    
    return amplitude, stretch

def gaussian_random_field(size, flag_normalize = True):
    """
        Input args:
            size (integer):
                The size of the cube output Gaussian random fields
            max_stretch (float):
                The maximum amount by which the fields will be stretched
                when training the CNN to identify AP-like stretch
            h (float):
                Hubble parameter
            flag_normalize (boolean, default = True):
                Normalizes the Gaussian Field:
                    - to have an average of 0.0
                    - to have a standard deviation of 1.0

        Returns:
            gfield (numpy array of shape (size, size, size)):
                The gaussian random field
                
        Example:
        import matplotlib.pyplot as plt
        example = gaussian_random_field(32, 1.1, 0.7)
        plt.imshow(example)
        """
 
        # Draws a complex gaussian random noise with normal (circular) distribution
    noise = np.random.normal(size = (size, size, size)) \
        + 1j * np.random.normal(size = (size, size, size))

    power, stretch = stretched_power()
    amplitude = np.sqrt(power)
    
        # To real space
    gfield = np.fft.ifftn(noise * amplitude).real
    
        # Sets the standard deviation to one
    if flag_normalize:
        gfield = gfield - np.mean(gfield)
        gfield = gfield/np.std(gfield)
        
    return gfield, stretch

def Gaussian_to_Galaxy_Field(N_gal):
    Gaussian_field, stretch = gaussian_random_field(field_size) #This gets us a 3D array with values representing a Gaussian random field with variance 1
    density_field_weights = np.exp(Gaussian_field - 1/2) #Takes Gaussian field and converts it to a realistic density field weighting
    sum_weights = np.sum(density_field_weights)
    galaxy_probs = density_field_weights / sum_weights
    
    # This uses the density field weighting and populates an array representing the actual distribution of galaxies in 3D space
    galaxy_probs_flat = galaxy_probs.flatten()
    N_gal_dist_flat = np.random.multinomial(N_gal, galaxy_probs_flat)
    N_gal_dist = N_gal_dist_flat.reshape((field_size, field_size, field_size))
    return N_gal_dist, stretch