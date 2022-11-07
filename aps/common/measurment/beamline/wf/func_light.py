'''
This file contains all the small functions needed for the program
'''
import numpy as np
import sys
import scipy.ndimage as sf


def write_h5(result_path, file_name, data_dict):
    ''' this function is used to save the variables in *args to hdf5 file
        args are in format: {'name': data}
    '''
    import h5py
    import os
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    with h5py.File(os.path.join(result_path, file_name+'.hdf5'), 'w') as f:
        for key_name in data_dict:
            f.create_dataset(key_name, data=data_dict[key_name], compression="gzip", compression_opts=9)
    prColor('result hdf5 file : {} saved'.format(file_name+'.hdf5'), 'green')


def read_h5(file_path, key_name, print_key=False):
    '''
        read the data with the key_name in the h5 file
    '''
    import h5py
    import os
    if not os.path.exists(file_path):
        prColor('Wrong file path: {}'.format(file_path), 'red')
        sys.exit()

    with h5py.File(file_path, 'r') as f:
        # List all groups
        if print_key:
            prColor("Keys: {}".format(list(f.keys())), 'green')

        # Get the data
        if isinstance(key_name, list):
            data = []
            for each_key in key_name:
                data.append(f[each_key][:])
        elif isinstance(key_name, str):
            data = f[key_name][:]
        else:
            prColor('Wrong h5 key name', 'red')
    return data


def write_json(result_path, file_name, data_dict):
    import os
    import json
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    file_name_para = os.path.join(result_path, file_name+'.json')
    with open(file_name_para, 'w') as fp:
        json.dump(data_dict, fp, indent=0)
    
    prColor('result json file : {} saved'.format(file_name+'.json'), 'green')


def read_json(filepath, print_para=False):
    import os
    import json
    if not os.path.exists(filepath):
        prColor('Wrong file path: {}'.format(filepath), 'red')
        sys.exit()
    # file_name_para = os.path.join(result_path, file_name+'.json')
    with open(filepath, 'r') as fp:
        data = json.load(fp)
        if print_para:
            prColor('parameters: {}'.format(data), 'green')
    
    return data


def noise_poisson(data, noise_level):
    '''
        here's function to add noise to the intensity
        input:
            data:               the original image
            noise_level:        the poisson noise level
        output:
            data_new:           the data with poisson noise
    '''
    noise = np.random.poisson(noise_level, data.shape)
    if data.dtype == 'complex128':
        I = np.abs(data)**2
        noisy_img = I+noise
        noisy_img = np.sqrt(noisy_img) * np.exp(1j*np.angle(data))
    else:
        noisy_img = data + noise
    return noisy_img


def spherical_wave(M, dx, beam_dia, L_focal, wavelength, amp_type):
    '''
    this function can generate spherical wavefront and amplitude
        M:              the matrix size
        dx:             pixel size
        beam_dia:       beam diameter, FWHM (gaussain) or whole size (plane)
        wavelength:     the wavelength of the light
        L_focal:        focal length of wavefront, negtive: diverged; positive: focused
        amp_type:       the amplitude initilization type, Gaussian or circle plane
    '''
    x = np.arange(-M/2, M/2) * dx
    y = np.arange(-M/2, M/2) * dx
    XX, YY = np.meshgrid(x, y)
    k = 2 * np.pi / wavelength

    phase = np.exp(-1j * k / (2 * L_focal) * (XX**2 + YY ** 2))

    if amp_type == 'Gaussian':
        # Gaussian amplitude distribution
        amp = np.exp(-(XX**2 + YY**2)/(beam_dia ** 2 / 4 / (np.log(2)))) * phase
    elif amp_type == 'plane':
        amp = np.ones(phase.shape, dtype=float) * (XX ** 2 + YY ** 2 < (beam_dia/2)**2) * phase
    else:
        sys.exit('wrong amplitude initialization type, Gaussian or plane')

    return amp


def phase_plate(M, dxy, period, dis_type, phase_value):
    '''
    To generate 2D phase distribution
        M:              array size
        dxy:            pixel size
        period:         the size of structure, for random phase plate, it's point size;
                        for grating, it's structure period
        dis_type:       random plate or 2D grating
        phase_value:    the max-min phase value of the distribution
    '''
    cluster_size_random = np.int64(round(period/dxy))
    cluster_size_grating = np.int64(round(period/2/dxy))
    # size of the initial matrix, then expand the matrix to make it larger than M

    if dis_type == 'random plate':
        '''
            use random plate as the initial matrix
        '''
        ini_size = int(M//cluster_size_random + 1)
        ini_array = np.random.rand(ini_size, ini_size)
        ini_array = np.ones(ini_array.shape) * (ini_array > 0.5)
        expande_array = np.repeat(np.repeat(ini_array, cluster_size_random, axis=0), cluster_size_random, axis=1)

    elif dis_type == '2D grating':
        '''
            use 2D grating as the initial matrix
        '''
        ini_size = int(M//cluster_size_grating + 1)
        ini_array_row = np.tile(np.arange(ini_size) % 2, (ini_size, 1))
        ini_array_col = np.transpose(ini_array_row)
        ini_array = ini_array_row * ini_array_col
        expande_array = np.repeat(np.repeat(ini_array, cluster_size_grating, axis=0), cluster_size_grating, axis=1)

    return np.exp(1j * phase_value * expande_array[0:M, 0:M])


def amp_plate(M, dxy, period, dis_type, absorption):
    '''
    To generate 2D amplitude distribution
        M:              array size
        dxy:            pixel size
        period:         the size of structure, for random phase plate, it's point size;
                        for grating, it's structure period
        dis_type:       random plate or 2D grating
        phase_value:    the max-min phase value of the distribution
    '''
    cluster_size_random = np.int64(round(period/dxy))
    cluster_size_grating = np.int64(round(period/2/dxy))
    # size of the initial matrix, then expand the matrix to make it larger than M

    if dis_type == 'random plate':
        '''
            use random plate as the initial matrix
        '''
        ini_size = int(M//cluster_size_random + 1)
        ini_array = np.random.rand(ini_size, ini_size)
        ini_array = np.ones(ini_array.shape) * (ini_array > 0.5)
        expande_array = np.repeat(np.repeat(ini_array, cluster_size_random, axis=0), cluster_size_random, axis=1)

    elif dis_type == '2D grating':
        '''
            use 2D grating as the initial matrix
        '''
        ini_size = int(M//cluster_size_grating + 1)
        ini_array_row = np.tile(np.arange(ini_size) % 2, (ini_size, 1))
        ini_array_col = np.transpose(ini_array_row)
        ini_array = ini_array_row * ini_array_col
        expande_array = np.repeat(np.repeat(ini_array, cluster_size_grating, axis=0), cluster_size_grating, axis=1)

    return 1 - absorption * expande_array[0:M, 0:M]

def image_roi(img, M):
    '''
        take out the interested area of the all data.
        input:
            img:            image data, 2D or 3D array
            M:              the interested array size
                            if M = 0, use the whole size of the data
        output:
            img_data:       the area of the data
    '''
    img_size = img.shape
    if M == 0:
        return img
    elif len(img_size) == 2:
        if M > min(img_size):
            return img
        else:
            pos_0 = np.arange(M) - np.round(M/2) + np.round(img_size[0]/2)
            pos_0 = pos_0.astype('int')
            pos_1 = np.arange(M) - np.round(M/2) + np.round(img_size[1]/2)
            pos_1 = pos_1.astype('int')
            img_data = img[pos_0[0]:pos_0[-1]+1, pos_1[0]:pos_1[-1]+1]
    elif len(img_size) == 3:
        if M > min(img_size[1:]):
            return img
        else:
            pos_0 = np.arange(M) - np.round(M/2) + np.round(img_size[1]/2)
            pos_0 = pos_0.astype('int')
            pos_1 = np.arange(M) - np.round(M/2) + np.round(img_size[2]/2)
            pos_1 = pos_1.astype('int')
            img_data = np.zeros((img_size[0], M, M))
            for kk, pp in enumerate(img):
                img_data[kk] = pp[pos_0[0]:pos_0[-1]+1, pos_1[0]:pos_1[-1]+1]

    return img_data


def prColor(word, color_type):
    ''' function to print color text in terminal
        input:
            word:           word to print
            color_type:     which color
                            'red', 'green', 'yellow'
                            'light_purple', 'purple'
                            'cyan', 'light_gray'
                            'black'
    '''
    end_c = '\033[00m'
    if color_type == 'red':
        start_c = '\033[91m'
    elif color_type == 'green':
        start_c = '\033[92m'
    elif color_type == 'yellow':
        start_c = '\033[93m'
    elif color_type == 'light_purple':
        start_c = '\033[94m'
    elif color_type == 'purple':
        start_c = '\033[95m'
    elif color_type == 'cyan':
        start_c = '\033[96m'
    elif color_type == 'light_gray':
        start_c = '\033[97m'
    elif color_type == 'black':
        start_c = '\033[98m'
    else:
        print('color not right')
        sys.exit()

    print(start_c + str(word) + end_c)


def normalize(data, b_min=0, b_max=1):
    '''
        this function is used to normalize the data to [0, 1]
        input:
            data:           the initial data
            b_min:          the minimum after normalization
            b_max:          the maximum after normalization
    '''
    return (data - np.amin(data)) / (np.amax(data) - np.amin(data)) * (b_max-b_min) + b_min

'''
    the 2D integration method here. fourier_integreation1 and 2 and frankotchellappa method
'''
def fourier_integration2DShift(dpc_x, del_x,
                               dpc_y, del_y):

    '''
    This function is the direct use of the CONTINOUS formulation of
    Frankot-Chellappa, eq 21 in the article:

    T. Frankot and R. Chellappa
        A Method for Enforcing Integrability in Shape from Shading Algorithms,
        IEEE Transactions On Pattern Analysis And Machine Intelligence, Vol 10,
        No 4, Jul 1988

    In addition, it uses the CONTINOUS shift property to avoid singularities
    at zero frequencies
    input:
        dpc_x:              the differential phase along x
        del_x:              pixel size along x
        dpc_y:              the differential phase along y 
        del_y:              pixel size along y         
    output:
        phi:                phase calculated from the dpc
    '''
    dpc_x = dpc_x / del_x
    dpc_y = dpc_y / del_y

    dim = dpc_x.shape
    x_axis = del_x * (np.arange(dim[1]) - round(dim[1]/2))
    y_axis = del_y * (np.arange(dim[0]) - round(dim[0]/2))

    xx, yy = np.meshgrid(x_axis, y_axis)

    dk_x = 2*np.pi / (dim[1]*del_x)
    dk_y = 2*np.pi / (dim[0]*del_y)

    kx = dk_x * (np.arange(dim[1]) - round(dim[1]/2))
    ky = dk_y * (np.arange(dim[0]) - round(dim[0]/2))

    fy, fx = np.meshgrid(ky/2/np.pi, kx/2/np.pi, indexing='ij')

    fft2 = lambda x: np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
    ifft2 = lambda x: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))
    # fftshift = lambda x: np.fft.fftshift(x)
    # ifftshift = lambda x: np.fft.ifftshift(x)

    # fx, fy = np.meshgrid(np.fft.fftfreq(dpc_x.shape[1], delx),
    #                      np.fft.fftfreq(dpc_x.shape[0], dely))

    # xx, yy = wpu.realcoordmatrix(dpc_x.shape[1], delx,
    #                              dpc_x.shape[0], dely)

    fo_x = np.abs(fx[0,1]*0.33) # shift fx value
    fo_y = np.abs(fy[1,0]*0.33) # shift fy value


    phaseShift = np.exp(2*np.pi*1j*(fo_x*xx + fo_y*yy))  # exp factor for shift

    mult_factor = 1/(2*np.pi*1j)/(fx - fo_x - 1j*fy + 1j*fo_y )


    bigGprime = fft2((dpc_x - 1j*dpc_y)*phaseShift)
    bigG = bigGprime*mult_factor

    phi = ifft2(bigG) / phaseShift

    phi -= np.min(np.real(phi))  # since the integral have and undefined constant,
                         # here it is applied an arbritary offset


    return phi


def gradient_nabla(phi):
    '''
        use fourier transfrom to calculate the displacement from the phase
        here phase is in a unit of [1/(k_wave)/p_x^2*z]
    '''
    fft2 = lambda x: np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
    ifft2 = lambda x: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))
    fftshift = lambda x: np.fft.fftshift(x)
    # ifftshift = lambda x: np.fft.ifftshift(x)
    
    NN, MM = phi.shape

    wx, wy = np.meshgrid(np.fft.fftfreq(MM)*2*np.pi,
                         np.fft.fftfreq(NN)*2*np.pi, indexing='xy')
    wx = fftshift(wx)
    wy = fftshift(wy)


    displace_x = np.real(ifft2(1j*wx*fft2(phi)))
    displace_y = np.real(ifft2(1j*wy*fft2(phi)))

    return [displace_x, displace_y]


def frankotchellappa(dpc_x,dpc_y):
    '''
        Frankt-Chellappa Algrotihm
        input:
            dpc_x:              the differential phase along x
            dpc_y:              the differential phase along y       
        output:
            phi:                phase calculated from the dpc
    '''
    fft2 = lambda x: np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
    ifft2 = lambda x: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))
    fftshift = lambda x: np.fft.fftshift(x)
    # ifftshift = lambda x: np.fft.ifftshift(x)
    
    NN, MM = dpc_x.shape

    wx, wy = np.meshgrid(np.fft.fftfreq(MM)*2*np.pi,
                         np.fft.fftfreq(NN)*2*np.pi, indexing='xy')
    wx = fftshift(wx)
    wy = fftshift(wy)
    numerator = -1j*wx*fft2(dpc_x) -1j*wy*fft2(dpc_y)
    # here use the numpy.fmax method to eliminate the zero point of the division
    denominator = np.fmax((wx)**2 + (wy)**2, np.finfo(float).eps)

    div = numerator/denominator


    phi = np.real(ifft2(div))

    phi -= np.mean(np.real(phi))

    return phi


'''
    here is the data pre-processing, normalization and the de-noise process.s
'''
def Denoise_rof(img, theta=1/8, n_iter=100, alpha=0.95):
    '''
        here use the Rudin-Osher-Fatemi method to reduce the noise.
        and then normalize the data
        input:
            img:                speckle image data
            theta:              parameter for the ROF method
            n_iter:             iteration for the ROF method
            alpha:              parameter for the ROF method
        output:
            img_out:            the processed image data
    '''
    dim = img.shape
    if len(dim) > 2:
        # if the data is a combination of multiple data, use loop to process
        n_data = dim[0]
        multi_flag = True
    else:
        n_data = 1
        multi_flag = False
    img_out = np.zeros(dim)
    # do the process of the data
    if not multi_flag:
        # normalize the data to [-1, 1]
        temp = normalize(img, -1, 1)
        temp_denoise = fista_l1_denoiseing(temp, theta, n_iter)
        img_out = normalize(temp - alpha * temp_denoise, 0, 255)

    elif multi_flag:
        # for multiple images
        for kk in range(n_data):
            # normalize the data to [-1, 1]
            temp = normalize(img[kk], -1, 1)
            temp_denoise = fista_l1_denoiseing(temp, theta, n_iter)
            img_out[kk] = normalize(temp - alpha * temp_denoise, 0, 255)
    return img_out


def fista_l1_denoiseing(data, theta, n_iter):
    '''
        Rudin-Osher-Fatemi method, L1 denoising
        input:
            data:           initial data, 2D array
            theta:          parameter
            n_iter:         iteration number
        output:
            data_new:       data after denoising
    '''
    bc = 'reflect'
    # axis=1 direction deviation
    nabla_x_kern = np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]])
    # axis=0 direction deviation
    nabla_y_kern = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]])
    
    # calculate the gradient along x, and y axis
    nabla = lambda x: np.array([sf.correlate(x, nabla_x_kern, mode=bc), sf.correlate(x, nabla_y_kern, mode=bc)])
    # calculate the nabla transpose
    nablaT = lambda x: sf.correlate(x[0], np.flip(nabla_x_kern), mode=bc) + sf.correlate(x[1], np.flip(nabla_y_kern), mode=bc)

    # define proximal operator of ||x||_{\infty} <= 1
    prox_linf = lambda x: np.fmin(np.fmax(x, -1), 1)
    # the objective
    obj = lambda x: np.sum(np.abs(nablaT(x) - data/theta)**2)
    # the step size
    tau = 1/4
    # start
    ini_data = np.zeros((2, data.shape[0], data.shape[1]))
    # gradient descent
    for kk in range(n_iter):
        # tt = obj(ini_data)
        # print('k={}, obj={:.2f}'.format(kk, tt))
        ini_data = prox_linf(ini_data - tau * nabla(nablaT(ini_data) - data/theta))
    print('data preprocess ends. obj={:.2f}'.format(obj(ini_data)))
    data_new = data - nablaT(ini_data) * theta
    return data_new


def grad_fourier(img, dxy=1, z=1, wavelength=2*np.pi):
    '''
        here use the fourier method to calculate the gradient along x and y
        here's the method: nabla(x) = ifft2(1j(kx, ky)fft2(x))
        if dxy, z, wavelength is not defined, it's the no unit gradient
        input:
            img:           the input intensity
            dxy:            pixel size
        output:
            D_y:            gradient along y or displacement
            D_x:            gradient along x or displacement
            
    '''
    dim = img.shape
    k_wave = 2 * np.pi / wavelength
    dk_x = 2*np.pi / (dim[1]*dxy)
    dk_y = 2*np.pi / (dim[0]*dxy)

    kx = dk_x * (np.arange(dim[1]) - round(dim[1]/2))
    ky = dk_y * (np.arange(dim[0]) - round(dim[0]/2))

    k_YY, k_XX = np.meshgrid(ky, kx, indexing='ij')
    fft2 = lambda x: np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
    ifft2 = lambda x: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))
    '''
        to calculate the diff_T:
            diff_T = 1j * F^(-1){(kx, ky) * F{}}
    '''
    
    nabla_T = lambda x: 1j * np.array([ifft2(k_YY * fft2(x)), ifft2(k_XX * fft2(x))])
   
    [D_y, D_x] = np.real(nabla_T(img * z / k_wave))
    return D_y, D_x


def grad_conv(img, method=2):
    '''
        use convolution to calculate the gradient
        method:             2 for O(h^4) and 1 for O(h2)
    '''
    if method == 2:
        h = np.array([[0, 0, 0, 0, 0], [1, -8, 0, 8, -1], [0, 0, 0, 0, 0]])/12
    elif method == 1:
        h = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])/2
    else:
        print('no defined method, should be 1 or 2 for grad_conv')
    
    wx = sf.correlate(img, h, mode='reflect')
    wy = sf.correlate(img, h.transpose(), mode='reflect')

    return wy, wx


def subpixel_shift(img, Sprow, Spcol):
    ''' input array: numpy.ndarray
        output array: numpy.ndarray
    '''
    if len(img.shape) > 2:
        print('subpixel_shift function cannot deal with 3D matrix')
        return
    N_row = img.shape[0]
    N_col = img.shape[1]
    if N_row % 2 != 0 or N_col % 2 != 0:
        print('The size of data is better to be 2^N for higher accuracy!')

    dk_row = 1/N_row
    dk_col = 1/N_col

    k_row = dk_row * np.fft.ifftshift(np.arange(-N_row/2, N_row/2) if N_row % 2 == 0 else np.arange(-(N_row-1)/2, (N_row+1)/2))
    k_col = dk_col * np.fft.ifftshift(np.arange(-N_col/2, N_col/2) if N_col % 2 == 0 else np.arange(-(N_col-1)/2, (N_col+1)/2))

    KK_row, KK_col = np.meshgrid(k_row, k_col)

    Fr = np.fft.fft2(np.fft.ifftshift(img))

    Fr = Fr * np.exp(-2j * np.pi * Spcol * KK_col) * np.exp(-2j * np.pi * Sprow * KK_row)
    output = np.fft.fftshift(np.fft.ifft2(Fr))

    return np.real(output)


def image_shift(img, Spy, Spx):
    '''
        here's the function to shift the images x and y pixels
        the pixel number can be sub-pixels
        input:
            img:            original data
            spy:            shift pixels along y axis
                            spy>0: down, spy<0: up
            spx:            shift pixels along x axis
                            spx>0: right, spx<0: left
    '''
    # integer part of the position shifting
    n_spy = np.round(Spy)
    n_spx = np.round(Spx)
    x_axis = (np.arange(img.shape[1]) + n_spx) % img.shape[1]
    x_axis = x_axis.astype('int')
    y_axis = (np.arange(img.shape[0]) + n_spy) % img.shape[0]
    y_axis = y_axis.astype('int')

    img_int = img[y_axis][x_axis] + 0

    s_y = Spy - n_spy
    s_x = Spx - n_spx

    img_new = subpixel_shift(img_int, s_y, s_x)

    return img_new


def image_align(image, offset_image):
    '''
        here's a function to do the alignment of two images.
        the offset_image is shifted relatively to image to find the best position
        and return the shifted back offset_image
        input:
            image:              the first image
            offset_image:       the second image
        output:
            pos:                best shift postion to maximize the correlation
            image_back:         the image after alignment
    '''
    from skimage.feature import register_translation
    from skimage.feature.register_translation import _upsampled_dft
    # from skimage.registration import phase_cross_correlation
    from scipy.ndimage import fourier_shift
    # roi = lambda x: x[0:100][0:100]
    shift, error, diffphase = register_translation(image, offset_image, 100)
    # shift, error, diffphase = phase_cross_correlation(image, offset_image, 100)
    # shift, error, diffphase = register_translation(roi(image), roi(offset_image), 100)

    print('shift dist: {}, alignment error: {} and phase difference: {}'.format(shift, error, diffphase))
    # image_back = image_shift(offset_image, shift[0], shift[1])
    image_back = fourier_shift(np.fft.fftn(offset_image), shift)
    image_back = np.real(np.fft.ifftn(image_back))
    return shift, image_back


def image_preprocess(image, have_dark, dark_img, have_flat, flat_img):
    '''
        do the flat or dark correction for the images
        img:            image to be corrected
        have_dark:      if there is dark
        dark_img:           dark image
        have_flat:      if there is flat
        flat_img:           flat image        
    '''
    if (have_flat != 0 and have_dark != 0):
        numerator = (flat_img - dark_img).clip(0.00000001)
        # numerator = numerator / np.amax(numerator)
        image = ((image - dark_img) / numerator ) * np.amax(image)
    elif (have_dark != 0):
        image = (image - dark_img).clip(0.00000001)
    elif (have_flat != 0):
        flat_img[flat_img == 0] = 0.00000001
        # flat_img = flat_img / np.amax(flat_img)
        image = (image / flat_img)  * np.amax(image)
                
    return image
