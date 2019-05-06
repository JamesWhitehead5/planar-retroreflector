import numpy as np
import numpy
from numpy import fft
import h5py
#import xarray



def _assert_2d(field):
    """Checks if the array is 2D"""
    assert len(np.shape(field))==2, "Variable fields must be a 2D array"

    
def propagate_angular(field, k, z_list, dx, dy):
    """Uses an angular propagation method to propagates a field to multiple planes 
    
    Parameters:
        field (2D array): Complex field array to be propagated.
        k (complex): The wavenumber of the propagation medium. Can be calculated using 2*pi*n/wavelength
        z_list (1D array): Distances that the field should be propagated
        dx (float): physical size of each pixel in the x direction
        dy (float): physical size of each pixel in the y direction
    
    Returns: 
        E_prop (3D array complex): Array of the propagated fields. 
            The 1st, 2nd and 3rd dimensions correspond to the x, y, and z dimensions.
            The third dimension is the same as the length of z_list
    """


    #Check parameter data types
    number_type = (float, int, complex)
    _assert_2d(field)
    assert isinstance(k, number_type), "k must be type float"
    assert all([isinstance(z, number_type) for z in z_list]), "z_list must be a 1D list"
    assert isinstance(dx, number_type), 'dx must be a number type'
    assert isinstance(dy, number_type), 'dy must be a number type'

    #shape of the input field
    n_x, n_y = field.shape

    #define spaxial frequency vectors
    k_x = np.fft.fftfreq(n=n_x, d=dx) *2*np.pi
    k_y = np.fft.fftfreq(n=n_y, d=dy) *2*np.pi
    k_Y, k_X = np.meshgrid(k_y, k_x, indexing='xy')
    
    #instantiate the total field matrix (3D)
    field_matrix = np.zeros(field.shape + (len(z_list),), dtype=complex)

    #fourier transform the input field
    E_in_k = np.fft.fft2(field)

    #define wavenumber for each wavevector in the direction of propagation
    k_z = np.sqrt(0j + k**2 - k_X**2 - k_Y**2)
    
    #broadcast E_in_k along the 3rd dimenstion
    E_k = np.repeat(E_in_k[:, :, np.newaxis], len(z_list), axis=2)
    
    #broadcast k_z into 3rd dimension
    k_Z = np.repeat(k_z[:, :, np.newaxis], len(z_list), axis=2)

    phase = k_Z*z_list
    E_k_prop = E_k * np.exp(1j*phase)
    E_prop = np.fft.ifft2(E_k_prop, axes=(0,1))
    
    return E_prop


def propagate_angular_padded(field, k, z_list, dx, dy):
    """Pads the field with n-1 zeros to remove boundary reflections"""
    
    _assert_2d(field)
    n_x, n_y = np.shape(field)
    pad_x = int(n_x)
    pad_y = int(n_y)
    
    padded_field = np.zeros(shape=(2*pad_x + n_x, 2*pad_y + n_y), dtype=complex)
    padded_field[pad_x:-pad_x, pad_y:-pad_y] = field

    E_prop_padded = propagate_angular(padded_field, k, z_list, dx, dy)
    E_prop = E_prop_padded[pad_x:-pad_x, pad_y:-pad_y, :]
    
    return E_prop

