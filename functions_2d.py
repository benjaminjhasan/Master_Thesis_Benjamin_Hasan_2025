import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from numba import njit, prange
from numba import jit
import time
import gc



    

# initialize bacteria

# The reason for placing b0 randomly, is that in real world experiments, we do not know what 'growth state' our cells are in.
    # That is, they could divide at any moment, I believe.

# initialize B

@njit
def init_b_2d(n, m, b0=1, p=10):
    
    grid_b = np.zeros((n, m, p), dtype=np.int32)  # (n, m, 10), initialize to zeros
    N = np.int32(n * m * b0)  # total number of cells to place
    n_sites = np.int32(n * m)
    if b0 > 1:
        print('b0 should be <= 1')
        b0 = np.float32(1)

    for i in range(n):
        for j in range(m):
            if np.random.rand() < (N / n_sites):
                k = np.random.randint(0, p, 1)          # generate 1 random index, between 0-10
                grid_b[i, j, k[0]] = np.int32(1)        # place a cell at that index
                N -= 1
            n_sites -= 1
    
    return grid_b


## ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ##


# initialize bacteria, nutrient, phages, and infected cells

@njit
def init_2d(n, m, b0, n0, i0, loc_i, p=10):
    
    # initialize grids
    grid_b = init_b_2d(n, m, b0, p)  # (n, m, 10), np.int32; bacteria, 1 per lattice site
    grid_n = np.ones((n, m), dtype=np.int32) *n0  # (n, m), np.int32; initialize to n0
    grid_p = np.zeros((n, m), dtype=np.int32)  # (n, m), np.int32; initialize to zeros
    grid_i = np.zeros((n, m, p), dtype=np.int32)    # (n, m, 10), np.int32; initialize to zeros
    
    # place infected cell
    grid_i[loc_i[0], loc_i[1], 0] = np.int32(i0)     # add infected cell at location loc_i, in state i=1

    return grid_b, grid_n, grid_p, grid_i            # np.int32, np.int32, np.int32, np.int32

## ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ##


# deterministic bacterial growth

@njit
def growth_det(b, n, dt, g_max=np.float32(2), n0=np.float32(3000)):

    K = np.float32(n0/5)                  # (, ), np.float32
    # g = np.float32(1)                     # (, ), np.float32      
    g = np.float32( g_max*n / (n+K) )   # (, ), np.float32      
    b += np.float32(g*b*dt)               # (, ), np.float32; if input is indeed np.float32. otherwise, np.float64
    n -= np.float32(g*b*dt)               # (, ), np.float32; if input is indeed np.float32
    if n<=0:
        n = np.float32(0)
    return b, n, g                        # np.float32, np.float32, np.float32


## ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ##

## ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ##

'Diffusion - not used currently'
# draw binomial numbers, 2D array (n, )

@njit(parallel=True)
def binomial_numba_2d(A, p0):
    
    n, m = np.int32(A.shape)
    jump = np.zeros((n, m), dtype=np.int32)
    jump_up = np.zeros((n, m), dtype=np.int32)
    jump_down = np.zeros((n, m), dtype=np.int32)
    jump_left = np.zeros((n, m), dtype=np.int32)
    jump_right = np.zeros((n, m), dtype=np.int32)
    
    # Iterate through each index
    for i in prange(n):                   
        for j in range(m):
            successes = np.int32(0)
            n_up = np.int32(0)
            n_down = np.int32(0)
            n_left = np.int32(0)
            n_right = np.int32(0)
            # Perform A[i, j] independent Bernoulli trials

            for k in range(A[i, j]):
                if np.random.rand() < p0[i, j]:
                    successes += np.int32(1)
                    rand = np.random.rand()
                    if 0 < rand <= np.float32(0.25):
                        n_up += np.int32(1)
                    if np.float32(0.25) < rand <= np.float32(0.5):
                        n_down += np.int32(1)
                    if np.float32(0.5) < rand <= np.float32(0.75):
                        n_left += np.int32(1)
                    if np.float32(0.75) < rand <= np.float32(1):
                        n_right += np.int32(1)


            jump[i, j] = successes
            jump_up[i, j] = n_up
            jump_down[i, j] = n_down
            jump_left[i, j] = n_left
            jump_right[i, j] = n_right

    return jump, jump_up, jump_down, jump_left, jump_right         # np.int32, asd, double check for all of them

## ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ##


'not in use currently' 
       # possible we get can get somethign wrong, due to parallelization, as per Seb. 


# Diffusion of particles 2D, binomial
@njit
def diffusion_binom(A_, dt, L, D):

    # print('D:', D)
    A = A_.astype(np.int32)         # (n, m), np.in32; copy inputs
    n, m = np.int32(A.shape)
    r = np.float32(4*D/(L**2))                                  # (, ), np.float32; probability of jumping
    p = np.ones((n, m), dtype=np.float32)* np.float32(r*dt)     # (n, m), np.float32; matrix, probability of jumping
    # if p.any() > 1:
    #     print('ERROR: p > 1')

    jump, j1, j2, j3, j4 = binomial_numba_2d(A, p)      # (n, m), np.int32

    # add diffusing partles
    A[1:, :] += j1[:-1, :]              # add particles moving down
    A[:-1, :] += j2[1:, :]              # add particles moving up
    A[:, 1:] += j3[:, :-1]              # add particles moving right
    A[:, :-1] += j4[:, 1:]              # add particles moving left
    
    # subtract the diffusing particles from origin
    A[:-1, :] -= j1[:-1, :]             # remove particles moving down
    A[1:, :] -= j2[1:, :]               # remove particles moving up
    A[:, :-1] -= j3[:, :-1]             # remove particles moving right
    A[:, 1:] -= j4[:, 1:]               # remove particles moving left

    return A        # (n, m), np.int32


## ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ##

'Rounding 2D array, given decimal values as our probabilities for rounding up'

@njit
def round_vec_2d(a_2d):
    decimal = a_2d - np.floor(a_2d)                                       # (n, m), np.float32; decimal part of a_2d. look out for floating point issues
    random = np.random.uniform(0, 1, size=a_2d.shape)                     # (n, m), np.float32; random numbers, shape of a_2d    
    random = random.astype(np.float32)                                    # np.float32
    a_2d = np.where(random < decimal, np.ceil(a_2d), np.floor(a_2d))      # (n, m), np.float32; round to nearest integer, according to random numbers
    return a_2d.astype(np.int32)

## ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ##

'Evaluate prob. matrix, 2D gaussian, for 1 particle to move to neighboring lattice sites, given D, dt and L'


@njit
def prob_jump(L, D, dt, rad_=25, n=25):        

    # size of prob. lattice
    l = np.int32(2*n + 1)           # length of square, defining prob. to jump
    p_square = np.zeros((l, l), dtype=np.float32)       # might wanna delete later
    p_circle = np.zeros((l, l), dtype=np.float32)       #  probabilities for 1 particle to jump
    center = np.int32(n)            # center of square
    radius = np.int32(rad_)         # radius of circle, in lattice sites: 10
    
    # evaluate p_circle
    for i in range(l):
        for j in range(l):
            p_square[i, j] = L**2 * 1/(4*np.pi*D*dt) * np.exp(-((i-center)*L)**2 /(4*D*dt)) * np.exp(-((j-center)*L)**2 /(4*D*dt))      
            
            if (i-center)**2 + (j-center)**2 <= radius**2:  # check if lattice site is within circle
                p_circle[i, j] = p_square[i, j]
            
    # renormalize
    p_circle = p_circle/np.sum(p_circle)        # renormalize circle asd 

    return p_circle

## ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ##

'Mirroring boundary, for diffusion of nutrient (and phages)'

@njit
def apply_mirroring_boundary(B, m, n, l):

    # Extract the inner part of B (A' before reflections)
    A_new = B[l:m+l, l:n+l].copy()                          # np.int32

    # Add mirrored corner regions
    A_new[:l, :l]     += B[:l, :l][::-1, ::-1]      # Top-left
    A_new[:l, -l:]    += B[:l, -l:][::-1, ::-1]     # Top-right
    A_new[-l:, :l]    += B[-l:, :l][::-1, ::-1]     # Bottom-left
    A_new[-l:, -l:]   += B[-l:, -l:][::-1, ::-1]    # Bottom-right

    # Add mirrored edge regions    
    A_new[:l, :]   += B[:l, l:-l][::-1, :]       # Top edge flipped down
    A_new[-l:, :]  += B[-l:, l:-l][::-1, :]      # Bottom edge flipped up
    A_new[:, :l]   += B[l:-l, :l][:, ::-1]       # Left edge flipped right
    A_new[:, -l:]  += B[l:-l, -l:][:, ::-1]      # Right edge flipped left

    return A_new

## ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ##

'Gaussian Diffusion, arbitrary dt'

@njit
def gauss_diffuse(A, prob_):
    # inputs A:           (m, n), np.int32

    grid_a = A.astype(np.float32)             # (n, m), np.float32; copy inputs
    prob = prob_.astype(np.float32)                     # (2l+1, 2l+1), np.float32; prob. to jump to neighbouring sites

    m, n = np.int32(grid_a.shape)           # np.int32, np.int32

    l = np.int32( (prob.shape[0]-1)/2)      # np.int32, our length l
    jump = np.zeros((n+2*l, m+2*l), dtype=np.int32)         # (n+2l, m+2l), np.int32; particles that jump, contains ghost nodes
    
    # loop
    for i in range(n):
        for j in range(m):
            jump_ij = grid_a[i, j] * prob               # (rad0+1, rad0+1), np.float32; floating point, no. particles that jump to neighbouring sites
            jump_ij = round_vec_2d(jump_ij)             # (m, n); round off number of particles that jump
            'here, evaluate a sort of redistribution of phages, so exact numbers are preserved? asd asd'
            jump[i:i+2*l+1, j:j+2*l+1] += jump_ij       # Add the particles that have jumped, onto extended lattice

    grid_a = apply_mirroring_boundary(jump, m, n, l)      # (m, n), np.int32; reflect particles back into original lattice

    return grid_a           # (m, n), np.int32

## ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ##

'Bacterial growth, 2D'

@njit
def growth_2d(B, ntr, dt, g_max=np.float32(2), n0=np.float32(3000), p=np.int32(10)):

    #copy input
    n, m, p = B.shape                   # n, m, p=10, np.int32
    grid_b = B.astype(np.int32)       # (n, m, 10), np.float32; bacteria
    grid_n = ntr.astype(np.float32)     # (n, m), np.float32  

    # asd: the cells jump through our 3D matrix, and once they go through the roof, they grow.
    g_mtrx = np.zeros((n, m, p), dtype=np.float32)                          # (n, m, p=10), np.float32; growth rate matrix
    lmd = np.zeros((n, m, p), dtype=np.float32)                             # (n, m, p=10), np.float32; growth prob. matrix
    jump = np.zeros((n, m, p), dtype=np.int32)                              # (n, m, p=10), np.int32; total number of cells that jump
    jump_roll = np.zeros((n, m, p), dtype=np.int32)                         # (n, m, p=10), np.int32; np.roll(1) jump matrix  

    'growth rate'          
    K = np.float32(n0/5)                  # (, ), np.float32; 'half-velocity constant', monod growth     
    g = p*g_max*grid_n/(grid_n + K)/( np.log(2))                      # (n, m), np.float64; growth rate / np.float64 without @njit
    g = g.astype(np.float32)                # (n, m), np.float32;
    # g = np.float32(1) * np.ones((n, m), dtype=np.float32)                      # (n, m), np.float32; growth rate / np.float64 without @njit

    'loop, growth'
    for i in range(n):
        for j in range (m):
            # for each lattice site, evaluate the whole p=10 column, so to speak. So we need g[i,j]
            g_mtrx[i, j, :] = g[i, j]  * np.ones(p, dtype=np.float32)                      # (p, ), np.float32; extend g from 1D to 2D
            lmd[i, j, :] = g_mtrx[i, j, :] * dt                                            # (p, ), np.float32; growth probability matrix

            for k in range(p):
                jump[i, j, k] = np.random.binomial(grid_b[i, j, k], lmd[i, j, k])          # (p, ), np.int32; number of cells that jump state
                jump_roll[i, j, k] = jump[i, j, (k - 1) % p]                               # (p, ), np.int32; custom np.roll(1, axis=2)
            jump_roll[i, j, 0] = jump[i, j, -1]                                            # (p, ), np.int32; growth, add last element to first element

    'add growth  + consume ntr'
    grid_b += jump_roll
    grid_b -= jump
    
    
    'for b0=0.05, max sum b = 155 000'  # check sum_b > 500 000 mb, dt=25
    threshold = np.float32(6000000)
    mask = np.sum(grid_b, axis=2) > threshold       #(m, n)
    index_0 = np.argwhere(mask == 1)                # (x, 2) ?

    # if sum_b exceeds threshold, set jump[x, y, -1] to 0
    if index_0.size > 0:
        for idx in index_0:
            jump[idx[0], idx[1], -1] = 0

    grid_b[:, :, 0] += jump[:, :, -1]                                 # (n, ), np.int32; add growth
    grid_n -= jump[:, :, -1]                                          # (n, ), np.int32; consume nutrients
    grid_n = np.where(grid_n < 0, np.int32(0), grid_n)                # np.int32; make sure n is non-negative. maybe delete later

    return grid_b, grid_n, np.sum(grid_b, axis=2), g*np.float32( np.log(2))/np.float32(10)     # np.int32, np.float32, np.float32, np.float32


## ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ##

## ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ##


# Distribute healthy infections

    # for 2D, by double indexing, e.g. i,j = (0,0), we access all the cells / growth states of colony (i,j) = (0,0)
@njit
def healthy_infections_2d(B, psn_):
    # grid_b, psn: (n, m, 10) + (n, m), np.int32 + np.int32; bacteria + healthy infection numbers

    grid_b = B.astype(np.int32)  # (n, m, 10), np.float32
    psn = psn_.astype(np.int32)  # (n, m), np.float32

    n = np.int32(grid_b.shape[0])                          # (, ), np.int32; number of colonies, n
    m = np.int32(grid_b.shape[1])                          # (, ), np.int32; number of colonies, m
    p = np.int32(grid_b.shape[2])                          # (, ), np.int32; number of colonies, p

    for i in range(n):
        for j in range(m):
            sum_colony = np.sum(grid_b[i, j])  # (, ), np.float32; sum of bacteria at lattice site i,j
            for k in range(p):                              # loop over growth states

                for _ in range(grid_b[i, j, k]):                               # loop over each element in grid_b[i, j, k]    
                    p_inf = np.float32 ( psn[i, j]/ sum_colony )                    # (, ), np.float32; probability of infection, for given cell
                    if p_inf > np.random.rand():                                    # if random number is smaller than p_inf
                        grid_b[i, j, k] -= np.int32(1)                                 # remove 1 cell, from state at grid_b[i, j, k]
                        psn[i, j] -= np.int32(1)                                       # remove 1 from psn[i]
                    sum_colony -= np.int32(1)                                          # remove 1 from sum_colony: proceed loop over the cells for that colony

    return grid_b


## ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ##


# Draw number of infections, from binomial distribution

# asd, add data types maybe?
    # and check for infections_drawn
@njit
def draw_infc_2d(P_, lmd):

    P = P_.astype(np.int32)
    n = P.shape[0]
    m = P.shape[1]

    infections_drawn = np.empty_like(P, dtype=np.int32)

    # # asd, delete later
    # if np.any(lmd > 1):
    #     print('\n', 'Error, Draw_infc_2d() : lmd > 1 !!', '\n')
    #     print('lmd max:', np.max(lmd))
    #     print('index where lmd > 1:', np.where(lmd > 1))
        
    for i in range(n):
        for j in range(m):
            infections_drawn[i, j] = np.random.binomial(P[i, j], lmd[i, j])        # Error, for large dt
            # infections_drawn[i, j] = np.int32(1)      # asd, delete later
    return infections_drawn

## ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ##


# Safe divide, 2D array
'mb add shapes below.'

@njit
def safe_divide(array1, array2):
    n = array1.shape[0]
    m = array1.shape[1]
    result = np.empty((n, m), dtype=np.float32)  # Output array to store the ratios
    
    for i in range(n):
        for j in range(m):
            # Check if array2[i] is zero or close to zero (to avoid NaN or Inf)
            if array2[i, j] > 1e-8:
                result[i, j] = array1[i, j] / array2[i, j]
            else:
                result[i, j] = 0.0 # Assign a default value (0) if division by zero

    return result       # (n, m), np.float32

## ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ##

# Apply mask, for 2D arrays

@njit
def apply_mask(arr1, arr2, arr3, ratio, mask_, b = np.float32(1)):

#    safe_div, ratio = apply_mask(safe_div, sum_i, n_surf, ratio, mask, b=1)       # (n, m), np.float32; apply mask, for colonies with sum_i > n_surf 
    'Applies the condition arr2 > arr3 and sets arr1 to 1 where the condition is true.'

    n, m = arr1.shape  # Get dimensions of the arrays
    
    for i in range(n):
        for j in range(m):
            if arr2[i, j] > arr3[i, j]:             # Check the condition, arr2 > arr3
                arr1[i, j] = np.float32(b)

            if mask_[i, j] == 1:                     # Check the condition, mask == 1
                ratio[i,j] = np.float32(1) - arr1[i, j]

    return arr1, ratio  # np.float32, np.float32


## ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ##


# shielding function

'theory'
# assume 1 cell as cross-sectional area 1 micrometer**3, and this is the area 1 cell can cover, of colony's total surface area.

    # V = (4/3) pi r**3
    # S = 4 pi r**2
    # r = ( 3*N/(4 pi) )**(1/3)

    # n surface = sum_tot - n_core
    # n_core = 4/3 / pi * (r-1)**3
    

@njit
def shielding_2d(B, I, delta_r=1):
    
    # print('delta_r:', delta_r)

    n = B.shape[0]
    m = B.shape[1] 
    ratio = np.zeros((n, m), dtype=np.float32)       # (n, m), np.float32; ratio of healthy cells to total cells, surface layer

    'check dimensions'
    # copy inputs
    grid_b = B.astype(np.float32)                # (n, m, p=10), np.float32; copy input
    grid_i = I.astype(np.float32)                # (n, m, p=10), np.float32; copy input
    sum_b = np.sum(grid_b, axis=2)               # (n, m), np.int32; sum of bacteria in each colony
    sum_b = sum_b.astype(np.float32)             # (n, m), np.float32; convert to float
    sum_i = np.sum(grid_i, axis=2)               # (n, m), np.int32; sum of infected cells in each colony
    sum_i = sum_i.astype(np.float32)             # (n, m), np.float32; convert to float
    # total cell count
    sum_tot = sum_b + sum_i                      # (n, m), np.float32; total number of cells in each colony, ~V
    sum_tot = sum_tot.astype(np.float32)         # (n, m), np.float32; convert to float

    # r + S, colonies
    r = (3/4 / np.pi * sum_tot)**(1/3)           # (n, m), np.float; radius of colonies
    r = r.astype(np.float32)                     # (n, m), np.float32
    # S_col = 4*np.pi*r**2                         # (n, m), np.float; surface area of colony
    # S_col = S_col.astype(np.float32)             # (n, m), np.float32

    # cells in surface layer
    if delta_r == 1:
        n_surf = 4*np.pi*r**2                               # (n, m), np.float32; assume thickness 1 micrometer, of surface layer
    else:
        'penetration depth, 2 micron:'
        n_surf = (4/3)*np.pi*(r**3 - (r - delta_r)**3)                               # (n, m), np.float32; assume thickness 2 micrometer, of surface layer   
    n_surf = round_vec_2d(n_surf)                # (n, m), np.int32; round to nearest integer, according to decimal part
    n_surf = n_surf.astype(np.float32)           # (n, m), np.float32; convert to float
    
    # where n_surf > sum_tot, set n_surf to sum_tot
    n_surf = np.where(n_surf > sum_tot, sum_tot, n_surf)  # (n, m), np.float32; if n_surf > sum_tot, set n_surf to sum_tot
    n_healthy = n_surf - sum_i                   # (n, m), np.float32; number of healthy cells at the surface
    n_healthy = np.where(n_healthy < 0, np.float32(0), n_healthy)   # in case cells in core also get infeced, I suppose
    
    n_core = sum_tot - n_surf                    # (n, m), np.float32; number of cells in core, asd dataype?
    mask = n_core > 0                            # (n, ), np.bool; mask, for colonies with core cells present

    # evaluate ratio, between sum_i, and n_surface cells
    ratio = safe_divide(sum_b, sum_tot)                 # (n, m), np.float32; ratio, if there are no cells in the core
    safe_div = safe_divide(sum_i, n_surf)               # (n, m), np.float32; ratio of infected cells to total cells, on surface
    # for safe_div, if sum_i > n_surf, set safe_div to 1

    # safe_div[sum_i > n_surf] = np.float32(1)            # (n, m) + (n, m), np.float32 + np.float32; if sum_i exceeds n_surf, safe_div makes it so ratio is set to 0 below
    safe_div, ratio = apply_mask(safe_div, sum_i, n_surf, ratio, mask, b=1)       # (n, m), np.float32; apply mask, for colonies with sum_i > n_surf

    # n_healthy = 0
    # n_surf = 0    
    return ratio, n_healthy, n_surf            # (n, m), np.float32 + np.float32 + np.float32; 

        # ratio of healthy cells to total cells, surface layer
        # number of healthy cells, surface layer
        # number of cells, surface layer


## ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ##



'asd: could be that ratio, n_surface, etc. are 1 timestep ahead of grid_b, etc. when you plot them.'


# Infections(), 2D

@njit
def infections_2d(B, N, P, I, dt, V_box, eta = np.float32(4*10**4), shielding=False, delta_r_ = 1):

    grid_b = B.astype(np.float32)     # (n, m, 10), np.float32; copy inputs 
    grid_n = N.astype(np.float32)     # (n, m), np.float32
    grid_p = P.astype(np.float32)     # (n, m), np.float32
    grid_i = I.astype(np.float32)     # (n, m, 10), np.float32
    # sum_b = np.sum(grid_b, axis = 2)                     # (n, ), np.float32; number of healthy bacteria, at each lattice site
    grid_total = np.sum(grid_b, axis = 2) + np.sum(grid_i, axis=2)            # (n, m), np.float32; combined infectible cells, healthy + infected

    # infections, evaluate
    eps = (grid_total)**(1/3)                              # (n, m), np.float32; total radius, as sum over number of cell radii
    eps = eps.astype(np.float32)               
    lmd = eta* eps* (1/V_box)* dt                        # (n, m), np.float32 / before np.float64; prob. infection, all bacterial cells
    lmd = lmd.astype(np.float32)

    # print('lmd:', lmd)
    infect0 = draw_infc_2d(grid_p, lmd)                     # (n, m), np.int32; no. infections, drawn from binomial distr.

    # ratio, infections
    'shielding'
    ratio, surf_hlthy, n_surf = shielding_2d(grid_b, grid_i, delta_r = delta_r_)                     # (n, m), np.float32; 
    'well-mixed / no shielding'
    if int(shielding) == 0:    # of no shielding
        ratio = safe_divide(np.sum(grid_b, axis=2), grid_total)               # (n, ), np.float32; ratio of infected cells to total cells, in surface layer
    
    infct2 = round_vec_2d(infect0 * ratio)                   # (m, n), np.int32; healthy infections, well-mixed model surface
    diff_infect = infect0 - infct2                        # (m, n), np.int32; difference between total infections and healthy infections

    # distribute healthy infections + add infectious cells + remove phages
    grid_b = healthy_infections_2d(grid_b, infct2)       # (n, 10), np.int32; distribute healthy infections randomly
    grid_p -= infect0                                   # (n, ), np.float32; remove phages 
    'asd asd, we should probably update ratio somewhere here, as it is used for lysis? Look at it.'
    
    grid_p = grid_p.astype(np.int32)                         # (n, ), np.int32; convert to int32
    grid_i = grid_i.astype(np.int32)                         # (n, 10), np.int32; convert to int32


    return grid_b, grid_p, grid_i, infct2, n_surf, ratio, diff_infect, surf_hlthy, np.float32(np.max(lmd)), infct2            # np.int32, np.float32, np.float32, np.int32


## ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ##

## ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ##

'Lysis'

## ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ##


# burst reinfection

@njit
def burst_reinf_2d(B, P, I, n_healthy, ratio, burst):

    grid_b = B.astype(np.float32)         # (n, m, 10), np.int32; copy inputs
    grid_p = P.astype(np.float32)       # (n, m), np.int32
    grid_i = I.astype(np.float32)       # (n, m, 10), np.int32

    n = grid_b.shape[0]                 # (, ), np.int32; number of colonies, n
    m = grid_b.shape[1]                 # (, ), np.int32; number of colonies, m
    p = grid_b.shape[2]                 # (, ), np.int32; number of colonies, p

    # reinfection
    grid_p += (burst/2).astype(np.float32)      # (n, m), np.float32; new free phages
    burst_inf = (burst/2)                       # (n, m), np.float64; number of infections from burst
    burst_inf = burst_inf.astype(np.int32)      # np.int32
    

    # round burst_inf using ratio; surface is well-mixed
    burst_inf = round_vec_2d(burst_inf * ratio)                         # (n, m), np.int32; round to nearest integer, according to decimal part
    burst_inf = np.where(n_healthy < burst_inf, n_healthy, burst_inf)   # (n, m), np.float64; restrict reinfections to healthy cells on surface
    burst_inf = burst_inf.astype(np.int32)      # np.int32
    grid_b = healthy_infections_2d(grid_b, burst_inf)                      # (n, m, 10), np.int32; distribute healthy infections randomly
    grid_i[:, :, 0] += burst_inf                                           # (n, m), np.float32; add new infections to grid_i

    grid_b = grid_b.astype(np.int32)
    grid_p = grid_p.astype(np.int32)                         # (n, m), np.int32; convert to int32
    grid_i = grid_i.astype(np.int32)                         # (n, m, 10), np.int32; convert to int32

    return grid_b, grid_p, grid_i           # np.int32, np.int32, np.int32

## ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ##

'Infectious States, Lysogeny'

@njit
def lysis_2d(B, P, I, tau_, dt, new_infect, n_healthy, ratio, g, n0=np.float32(3000), beta=np.int32(50), reinfection=False):
    
    grid_b = B.astype(np.float32)           # (m, n, 10) + (m, n, ) + (m, n, 10),  + np.float32 + np.float32; copy inputs
    grid_p = P.astype(np.float32)
    # grid_i = I.astype(np.float32)         
    grid_i = I.astype(np.int32)         
    n = grid_i.shape[0]                     # (, ), np.int32; number of colonies, x
    m = grid_i.shape[1]                     # (, ), np.int32; number of colonies, y
    p = grid_i.shape[2]                     # (, ), np.int32; number of inf. states

    'delayed lysis'
    # Test also without LIN, below.
    'work work: if g = 0, we need to set a value for tau_1'
    tau_1 = np.ones((n, m), dtype=np.float32) * tau_ / (g)               # (n, m), np.float32; latency time lysis, tied to nutrient availability
    # 'asd, error: try below safe divide, if it works.'
    # tau_1 = np.ones((m, n), dtype=np.float32) * safe_divide(tau_, g)               # (n, m), np.float32; latency time lysis, tied to nutrient availability
    # tau_1 = np.where(tau_1 == 0, np.float32(1000), tau_1)               # (n, m), np.float32; if 0, set to 1000h, i.e. for g = 0

    jump = np.zeros((m, m, p), dtype=np.int32)                           # (n, m, p=10), np.int32; total number of cells that jump
    jump_roll = np.zeros((m, n, p), dtype=np.int32)                      # (n, m, p=10), np.int32; np.roll(1) jump matrix  
    
    tau = np.zeros((m, n, 10), dtype=np.float32)                         # (n, m, 10), np.float32; latency time lysis
    for i in range(n):
        for j in range(m):
            tau[i,j] = tau_1[i,j] * np.ones(10, dtype=np.float32)              # (n, m, 10), np.float32; extend tau from 2D to '3D'

    # prob. to move inf. state,     'tau diverges and lmd==0'
    lmd = (10/tau)*dt                                       # (n, m, 10), np.float32; prob. for 1 cell to move one inf. state: 10/tau

        # asd: can we cut 1 loop below, as lmd is the same for all sites, and all states? Maybe not, as numba doesnt do np.binomial for more than 1 site at a time...
    for i in range(n):
        for j in range(m):
            for k in range(p):
                jump[i, j, k] = np.random.binomial(grid_i[i, j, k], lmd[i, j, k])         # (p, ), np.int32; number of cells that jump state
                jump_roll[i, j, k] = jump[i, j, (k - 1) % p]                              # (p, ), np.int32; custom np.roll(1, axis=2)
            jump_roll[i, j, 0] = jump[i, j, -1]                            # (p, ), np.int32; growth, add last element to first element

    'move infected cells'                
    grid_i = grid_i - jump + jump_roll           # (n, m, 10), np.float32; move cells between inf. states
    grid_i[:, :, 0] -= jump[:, :, -1]            # (n, m), np.float32; ensure cells that have lysed, are not added to state k=1
    grid_i[:, :, 0] += new_infect                # (n, m), np.float32 + np.int32; 1st column, add new infections, after 'jumps' have been made

    'lysis burst'
    burst = beta*jump[:, :, -1]         # (n, m), np.int32; lysis burst size 
    
    'no reinfection'
    if reinfection == False:
        # print('reinfction == False')
        grid_p += burst

        grid_b = grid_b.astype(np.int32)              # np.int32
        grid_p = grid_p.astype(np.int32)              # np.int32
        grid_i = grid_i.astype(np.int32)              # np.int32

        return grid_b, grid_p, grid_i, tau_1, jump[:, :, -1]                   # np.float32 + np.float32 + (n, ), np.float32 + (n, ), np.int32
    
    'if reinfection'
    # print('reinfction == True')
    grid_b, grid_p, grid_i = burst_reinf_2d(grid_b, grid_p, grid_i, n_healthy, ratio, burst)   # np.int32, np.float32, np.float32
    
    grid_b = grid_b.astype(np.int32)              # np.int32
    grid_p = grid_p.astype(np.int32)              # np.int32
    grid_i = grid_i.astype(np.int32)              # np.int32

    return grid_b, grid_p, grid_i, tau_1, jump[:, :, -1]                   # np.float32 + np.float32 + (n, ), np.float32 + (n, ), np.int32
    


## ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ##

'evalaute radial densities'

@njit
def radial_density(list_sumb, Lx):
    # evaluate radial density

    m, n = np.int32(list_sumb.shape)     # np.int32
    center = np.int32 ((m//2, n//2))        # np.int32

    # Define the maximum radius and thickness of concentric circles
    max_radius = np.float32 (np.sqrt((m // 2)**2 + (n // 2)**2) * Lx)            # np.float32; max radius in micrometer
    circle_thickness = np.float32(Lx)           # np.float32; 

    # Create arrays to store average densities
    num_circles = np.int32(max_radius // circle_thickness) + 1          # np.int32; +1, so we get whatever is left over
    average_densities = np.zeros(num_circles, dtype=np.float32)         # (num_circles, ) np.float32; densities
    av_std = np.zeros(num_circles, dtype=np.float32)                    # (num_circles, ) np.float32; standard deviation
    av_std_mean = np.zeros(num_circles, dtype=np.float32)               # (num_circles, ) np.float32; standard deviation / mean
    average_b = np.zeros(num_circles, dtype=np.float32)                 # (num_circles, ) np.float32; absolute bacterial count
    density_lattice = np.zeros((m, n), dtype=np.float32)                # np.float32, (m, n); store density for each lattice site
    std_lattice = np.zeros((m, n), dtype=np.float32)                    # np.float32, (m, n); store std for each lattice site
    std_mean_lattice = np.zeros((m, n), dtype=np.float32)               # np.float32, (m, n); store std/mean for each lattice site

    # Loop through each lattice site and calculate its distance from the center
    radius_matrix = np.zeros((m, n), dtype=np.float32)          # np.float32; store radiuses
    for i in range(m):
        for j in range(n):
            radius = np.float32 (np.sqrt((i - center[0])**2 + (j - center[1])**2) * Lx)
            radius_matrix[i, j] = radius                        # np.float32, (m, n); store radius, for all lattice sites
    
    
    # Calculate average densities for each concentric circle
    for k in range(num_circles):
        lower_bound = np.float32 (k * circle_thickness)             # np.float32
        upper_bound = np.float32 ((k + 1) * circle_thickness)       # np.float32
        

        # Find lattice sites within the current circle
        mask = (radius_matrix >= lower_bound) & (radius_matrix < upper_bound)       # np.bool_, (m, n); which sites i,j are within circle k
        
        # Calculate the average density for the current circle
        if np.sum(mask) > 0:  # Avoid division by zero
            n_k = np.int32(0)               # number of lattice sites within circle k
            sum_k = np.float32(0)           # sum over sum_b, for lattices within circle k
            
            list_k_sumb = np.zeros((m*n), dtype=np.float32)          # (m*n, ), np.float32; store sum_b for each lattice site within circle k
            count = np.int32(0)             # np.int32; counter for list_k_sumb

            for i in range(m):
                for j in range(n):
                    if mask[i, j]:
                        n_k += np.int32(1)
                        sum_k += list_sumb[i, j]
                        list_k_sumb[count] = list_sumb[i, j]
                        count += np.int32(1)
                        if count > m*n:
                            print('count:', count)
                            print('error, count > m*n')
                            break
            
            list_k_sumb = list_k_sumb[:count]                   # np.float32; remove zeros from list_k_sumb
            
            average_b[k] = sum_k / n_k                          # average absolute number of cells, for circle k
            average_densities[k] = ( sum_k ) / (n_k * Lx**2)     # evaluate density per area, as [ 1/ micrometer**2 ]
            # print('list_k_sumb:', list_k_sumb)
            av_std[k] = np.std(list_k_sumb)                     # (no. circles, ), np.float32; standard deviation for circle k
            if np.mean(list_k_sumb) == 0:
                av_std_mean[k] = np.float32(0)
            else:
                av_std_mean[k] = np.std(list_k_sumb) / np.mean(list_k_sumb)     # (no. circles, ), np.float23; standard deviation / mean for circle k

        else:                                   # if no lattice sites within circle k
            average_b[k] = np.float32(0)
            average_densities[k] = np.float32(0)
            av_std[k] = np.float32(0)
            av_std_mean[k] = np.float32(0)

        
        # store density for each lattice site
        for i in range(m):
            for j in range(n):
                if mask[i, j]:
                    density_lattice[i, j] = average_densities[k]         # np.float32, (m, n); store density for each lattice site
                    std_lattice[i, j] = av_std[k]                        # np.float32, (m, n); store std for each lattice site
                    std_mean_lattice[i, j] = av_std_mean[k]              # np.float32, (m, n); store std/mean for each lattice site

    'we return:'
        # density per area (no. circles, );     abs. numbers (no. circles, ); 
        # std per lattice site (m, n);          std/mean per lattice site (m, n);
        # density per lattice site (m, n);      radius for each lattice site (m, n)
    return average_densities, average_b, av_std, av_std_mean, std_lattice, std_mean_lattice, density_lattice, radius_matrix           # (num_circles, ), np.float32; (m, n), np.float32; and dunno

## ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ##

'Evaluate radial densities, std/mean, etc., for 1/4 system'

@njit
def radial_density_corner(list_var, L):
    # (m, n), np.int32;  (), np.float32
        # evaluate radial density

    m, n = np.int32(list_var.shape)
    
    # max radius, thickness of circles, no. circles
    max_radius = np.float32 (np.sqrt((m)**2 + (n)**2) * L)            # np.float32; max radius in micrometer
    circle_thickness = np.float32(L)           # np.float32; 
    num_circles = np.int32(max_radius // circle_thickness) + 1          # np.int32; +1, so we get whatever is left over

    # store average densities
    average_densities = np.zeros(num_circles, dtype=np.float32)         # (num_circ, ) np.float32; densities
    av_std = np.zeros(num_circles, dtype=np.float32)                    # (num_circ, ) np.float32; standard deviation
    av_std_mean = np.zeros(num_circles, dtype=np.float32)               # (num_circ, ) np.float32; standard deviation / mean
    average_b = np.zeros(num_circles, dtype=np.float32)                 # (num_circ, ) np.float32; absolute bacterial count
    density_lattice = np.zeros((m, n), dtype=np.float32)                # np.float32, (m, n); store density for each lattice site
    std_lattice = np.zeros((m, n), dtype=np.float32)                    # np.float32, (m, n); store std for each lattice site
    std_mean_lattice = np.zeros((m, n), dtype=np.float32)               # np.float32, (m, n); store std/mean for each lattice site
    
    # Loop through each lattice site and calculate its distance from the corner
    radius_matrix = np.zeros((m, n), dtype=np.float32)          # np.float32; store radiuses
    for i in range(m):
        for j in range(n):
            radius = np.float32 (np.sqrt(i**2 + j**2) * L)
            radius_matrix[i, j] = radius                        # np.float32, (m, n); store radius, for all lattice sites
    
    # Calculate average densities for each concentric circle
    for k in range(num_circles):
        lower_bound = np.float32 (k * circle_thickness)             # np.float32
        upper_bound = np.float32 ((k + 1) * circle_thickness)       # np.float32
    
        # Find lattice sites within the current circle
        mask = (radius_matrix >= lower_bound) & (radius_matrix < upper_bound)       # np.bool_, (m, n); which sites i,j are within circle k

        # Calculate the average density for the current circle
        if np.sum(mask) > 0:  # Avoid division by zero
            n_k = np.int32(0)               # np.int32; number of lattice sites within circle k
            sum_k = np.float32(0)           # np.float32; sum over sum_b, for lattices within circle k

            list_k_sumb = np.zeros((m*n), dtype=np.float32)          # (m*n, ), np.float32; store sum_b for each lattice site within circle k
            count = np.int32(0)             # np.int32; counter for list_k_sumb
        
            for i in range(m):
                for j in range(n):
                    if mask[i, j]:
                        n_k += np.int32(1)
                        sum_k += list_var[i, j]
                        list_k_sumb[count] = list_var[i, j]         # asd asd, change variable name? starting above...
                        count += np.int32(1)
                        if count > m*n:
                            print('count:', count)
                            print('error, count > m*n')
                            break
    
            list_k_sumb = list_k_sumb[:count]                   # np.float32; remove zeros from list_k_sumb
                
            average_b[k] = sum_k / n_k                          # average absolute number of cells, for circle k
            average_densities[k] = ( sum_k ) / (n_k * L**2)     # evaluate density per area, as [ 1/ micrometer**2 ]
            av_std[k] = np.std(list_k_sumb)                     # (no. circles, ), np.float32; standard deviation for circle k
            if np.mean(list_k_sumb) == 0:
                av_std_mean[k] = np.float32(0)
            else:
                av_std_mean[k] = np.std(list_k_sumb) / np.mean(list_k_sumb)     # (no. circles, ), np.float23; standard deviation / mean for circle k


        else:                                   # if no lattice sites within circle k
            average_b[k] = np.float32(0)
            average_densities[k] = np.float32(0)
            av_std[k] = np.float32(0)
            av_std_mean[k] = np.float32(0)

        # store density for each lattice site
        for i in range(m):
            for j in range(n):
                if mask[i, j]:
                    density_lattice[i, j] = average_densities[k]         # np.float32, (m, n); store density for each lattice site
                    std_lattice[i, j] = av_std[k]                        # np.float32, (m, n); store std for each lattice site
                    std_mean_lattice[i, j] = av_std_mean[k]              # np.float32, (m, n); store std/mean for each lattice site

    'we return:'
        # density per area (no. circles, );     abs. numbers (no. circles, ); 
        # std per lattice site (m, n);          std/mean per lattice site (m, n);
        # density per lattice site (m, n);      radius for each lattice site (m, n)

    return average_densities, average_b, av_std, av_std_mean, std_lattice, std_mean_lattice, density_lattice, radius_matrix           # (num_circles, ), np.float32; (m, n), np.float32; and dunno

## ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ##

'Evalute r_half, 1/4 system'

@njit
def calc_r_half_corner(sumb_, phg_, sumi_, L):

    grid_sumb = sumb_.astype(np.float32)        # (m, n), np.float32; copy of sum B, at last time step
    grid_p = phg_.astype(np.float32)            # (m, n), np.float32; copy of P, at last time step
    grid_sumi = sumi_.astype(np.float32)        # (m, n), np.float32; copy of sum I, at last time step

    # evaluate densities
    density_b, b_k, av_stdb, av_std_meanb, std_ijb, std_mean_ijb, density_lattice_b, _ = radial_density_corner(grid_sumb, L)  # bacteria
    density_p, phg_k, av_stdp, av_std_meanp, std_ijp, std_mean_ijp, density_lattice_p, _ = radial_density_corner(grid_p, L)     # phage
    density_i, i_k, av_stdi, av_std_meani, std_iji, std_mean_iji, density_lattice_i, _ = radial_density_corner(grid_sumi, L)  # infected

    # no phage presence
    indx_no_phg = np.argwhere(phg_k == 0)            # (k, ); find lattice site, where phg is not present, for b_k, our radial bacterial count
    indx_no_phg = indx_no_phg.T                      # 
    indx_no_phg = indx_no_phg.flatten()              # flatten array

    new_bk = b_k[indx_no_phg] + i_k[indx_no_phg]     # (k, ); radial bacterial count, where no phages are present
    mean_new_bk = np.mean(new_bk)                    # np.float32, mean radial bacterial count, where no phages are present

    # add b + i
    total_k = b_k + i_k         # (r, ), np.float32; sum of bacteria and infected bacteria, radial
    r_half = np.argwhere(total_k > mean_new_bk/2)       # (q, 1); find lattice site, where total_k > mean_new_bk
    r_half = r_half.flatten()                         # flatten array
    r_half = r_half.astype(np.int32)                  # np.int32

    plaque_indx = r_half[0]         # the plaque starts, where sum_b is first above 1/2 of mean sum_b, at t_end
    plaque_size = np.float32(plaque_indx * L)       # asd asd, consider factor before Lx, if we change circle thickness

    return plaque_size          # np.float32, plaque size in micrometer

## ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ##


## ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ##

'function, evaluating rad0p, etc., given dt'

def eval_p_circ(dt_):

    # might wanna remove np.float32 below, or we might get errors due to the data type not matching...
    if dt_ == np.float32(5/3600):
        rad0p = 5
        n0_circp = 5
        rad0n = 12          # prob. jump ntr
        n0_circn = 12
        
    if dt_ == np.float32(14/3600):
        rad0p = 7           # prob. jump phg
        n0_circp = 7
        rad0n = 18          # prob. jump ntr
        n0_circn = 18

    if dt_ == np.float32(20/3600):
        print('dt:', dt_)
        rad0p = 8           # prob. jump phg
        n0_circp = 8
        rad0n = 21          # prob. jump ntr
        n0_circn = 21

    if dt_ == np.float32(25/3600):
        rad0p = 9           # prob. jump phg
        n0_circp = 9
        rad0n = 22          # prob. jump ntr
        n0_circn = 22

    if dt_ == np.float32(30/3600):
        rad0p = 10           # prob. jump phg
        n0_circp = 10
        rad0n = 24          # prob. jump ntr
        n0_circn = 24

    if dt_ == np.float32(35/3600):
        rad0p = 10           # prob. jump phg
        n0_circp = 10
        rad0n = 28          # prob. jump ntr
        n0_circn = 28

    if dt_ == np.float32(40/3600):
        rad0p = 11           # prob. jump phg
        n0_circp = 11
        rad0n = 32          # prob. jump ntr
        n0_circn = 32
    
    if dt_ == np.float32(45/3600):
        rad0p = 12           # prob. jump phg
        n0_circp = 12
        rad0n = 33          # prob. jump ntr
        n0_circn = 33
    
    if dt_ == np.float32(50/3600):
        rad0p = 13           # prob. jump phg
        n0_circp = 13
        rad0n = 33          # prob. jump ntr
        n0_circn = 33

    if dt_ == np.float32(55/3600):
        rad0p = 13           # prob. jump phg
        n0_circp = 13
        rad0n = 34          # prob. jump ntr
        n0_circn = 34

    if dt_ == np.float32(60/3600):
        rad0p = 14           # prob. jump phg
        n0_circp = 14
        rad0n = 34          # prob. jump ntr
        n0_circn = 34

    return rad0p, n0_circp, rad0n, n0_circn


## ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ##


## ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ##

'Update function'

# Consider that you should ideally only run 1 for-loop across your bacterial grid, one over phages, and one over nutrients, plus one over grid_i.
    # unless they are updated multiple times, in between looping functions


'asd: inspect datatypes below, in between functions. Try to make output be np.int32, always. So, only use np.float32 when necessary for calculations.'

@njit   
def update(B, ntr, P, I, dt, V_box, eta, Lx, circ_p, circ_n, tau, n0_, shielding_, reinfection_, delta_r=1):
    # (m, n, 10) + (m, n, ) + (m, n, ) + (m, n, 10), np.int32; 
    
    grid_b = B.astype(np.int32)     # copy inputs
    grid_n = ntr.astype(np.int32)
    grid_p = P.astype(np.int32)
    grid_i = I.astype(np.int32)     

    'infections + diff. phg'
    grid_b, grid_p, grid_i, inf_new, n_surf, ratio, diff_infect, n_healthy, lmd, inf_healthy = infections_2d(grid_b, grid_n, grid_p, grid_i, dt, V_box, eta, shielding=shielding_, delta_r_ = delta_r)           # also returns n_surf, ratio, diff_infect
    grid_p = gauss_diffuse(grid_p, circ_p)           # (n, ), np.int32; diffusion of phages

    'growth + consump. + diff. ntr'
    grid_b, grid_n, sum_b, g = growth_2d(grid_b, grid_n, dt, n0=n0_)           # 
    grid_n = gauss_diffuse(grid_n, circ_n)           # (n, ), np.int32; diffusion of nutrients

    'infectious states'
    grid_b, grid_p, grid_i, tau1, lysis_no = lysis_2d(grid_b, grid_p, grid_i, tau, dt, inf_new, n_healthy, ratio, g, reinfection=reinfection_)
    
    return grid_b, grid_n, grid_p, grid_i, sum_b, lmd, diff_infect, n_healthy, ratio, lysis_no, inf_healthy   # np.int32, np.int32, np.int32, np.int32, np.int32

## ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ##
