# -------------------- Modules and Packages--------------------

# import scipy.constants as sc
import math as math
from numba import njit
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
from scipy.special import spherical_jn, spherical_yn


@njit
def V(r, g, mu): #r = radius, g = coupling strength, mu = screening
    '''
    r = radius
    g = coupling strength
    mu = screening (how fast potential dies)
    '''
    if r == 0.0:
        return 0.0   # avoid division by zero
    return -g * np.exp(-mu * r) / r

    
@njit
def k(E,r,g,mu,m,hbar):
    return np.sqrt(2 * m * (E - V(r,g,mu))) / hbar
    

@njit
def k_func(m, E, l, r, g, mu, hbar,flag=False):

    inv_hbar2 = 1.0 / (hbar * hbar)
    if flag:
        return (2.0*m*E)*inv_hbar2 - (l*(l+1.0))/(r*r) # zero potential
    else:
        return (2.0*m*E)*inv_hbar2 - (l*(l+1.0))/(r*r) - (2.0*m*V(r,g,mu))*inv_hbar2

    

@njit
def Numerov(l, E, rmax, g, h, mu, m, hbar,flag=False):
    rvals = np.arange(h, rmax, h) # avoid start at zero
    n = len(rvals)
    y = np.zeros(n)

    y[0] = 0.0
    y[1] = h ** (l + 1)

    h2 = h * h
    c = h2 / 12.0

    for i in range(1, n - 1):
        km1 = k_func(m, E, l, rvals[i - 1], g,mu, hbar,flag)
        kn  = k_func(m, E, l, rvals[i],     g,mu, hbar,flag)
        kp1 = k_func(m, E, l, rvals[i + 1], g,mu, hbar,flag)

        y[i + 1] = (2.0 * (1.0 - 5.0*c*kn) * y[i] - (1.0 + c*km1) * y[i - 1]) / (1.0 + c*kp1)

    return y, rvals



# -------------------- Functions for getting r and u values starting from outside the potential --------------------

@njit
def outside_vals(rvals, uvals,mu,g):
    index = 0
    n = len(rvals)

    for i in range(n):
        if abs(V(rvals[i],g, mu) ) < 1e-5:
            index = i
            break

    return uvals[index:], rvals[index:]

@njit
def r_1halfr_2(r, u, max_points=2):
    n = len(u)

    r_out = np.empty(max_points)
    u_out = np.empty(max_points)
    count = 0

    for i in range(1, n - 1):
        if (u[i] > u[i-1] and u[i] > u[i+1]) or (u[i] < u[i-1] and u[i] < u[i+1]):
            r_out[count] = r[i]
            u_out[count] = u[i]
            count += 1
            if count == max_points:
                break

    return r_out[:count], u_out[:count]

# -------------------- Functions for phase shifts (delta) and total cross section (sigma) --------------------
@njit
def K(rvals, uvals):
    """
    rvals here are starting from when there is no potential r > R

    The l value the uvals possesses will determine the l value/subscript
    that the phase shift delta_l will have.
    """
    K_array = []
    for i in range(len(rvals) - 1):
        K = (rvals[i] * uvals[i + 1]) / (rvals[i + 1] * uvals[i])
        K_array.append(K)
    return K_array


@njit
def AnalyticalDelta():
    return 5


def delta_l_notnumba(l, rvals, kvals, E):
    """
    rvals here are starting from when there is no potential r > R
    """

    deltavals = []
    k_0 = math.sqrt(2*E)

    for i in range(len(rvals) - 1):
        j_l_i = spherical_jn(l, k_0 * rvals[i])
        n_l_i = spherical_yn(l, k_0 * rvals[i])

        j_l_ip1 = spherical_jn(l, k_0 * rvals[i + 1])
        n_l_ip1 = spherical_yn(l, k_0 * rvals[i + 1])

        numerator = kvals[i] * j_l_i - j_l_ip1
        denominator = kvals[i] * n_l_i - n_l_ip1

        delta_i = np.arctan(numerator / denominator)

        deltavals.append(delta_i)
    return deltavals

@njit
def delta_l(l, rvals, kvals, E):
    """
    l=0 only.

    Uses:
      j0(x) = sin(x)/x
      n0(x) = -cos(x)/x
    """
    deltavals = []
    k_0 = math.sqrt(2.0 * E)

    for i in range(len(rvals) - 1):
        x_i   = k_0 * rvals[i]
        x_ip1 = k_0 * rvals[i + 1]

        if x_i == 0.0 or x_ip1 == 0.0:
            deltavals.append(0.0)
            continue

        j0_i   = math.sin(x_i)   / x_i
        n0_i   = -math.cos(x_i)  / x_i
        j0_ip1 = math.sin(x_ip1) / x_ip1
        n0_ip1 = -math.cos(x_ip1)/ x_ip1

        numerator   = kvals[i] * j0_i - j0_ip1
        denominator = kvals[i] * n0_i - n0_ip1

        delta_i = np.arctan(numerator / denominator)

        deltavals.append(delta_i)

    return deltavals



@njit
def ScatterLength(l: int,E,rmax, h,mu,Gmin,Gmax,N):

    gvals = np.linspace(Gmin,Gmax,N)
    delta = []
    k = np.sqrt(2 * E) # 2mE/hbar however, hbar = m = 1
    #sigmas = []

    for g_i in gvals:

        u,r = Numerov(l,E,rmax,g_i,h,mu,m=1,hbar=1)

        u_aug, r_aug = outside_vals(r, u ,mu,g_i)
        r_new, u_new = r_1halfr_2(r_aug, u_aug) 

        phaseshift = delta_l(l, r_new, K(r_new, u_new), E) 
        delta.append(phaseshift[-1])

    phase = np.array(delta)
    scattering_length = - phase / k
    return delta, scattering_length, gvals


# ============================ BLDMC ============================

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# ============================ FUNCTIONS ============================

@njit
def HistogramSum(H):
    '''
    Sums each bin in a histogram, NOTE: it returns the absolute value of each bin summed.
    '''
    s = 0.0
    for i in range(H.shape[0]):
        s += abs(H[i])
    return s

@njit
def get_bin(q, deltaq):
    return int(q // deltaq)

@njit
def u_0(potential):
    return (2.0 * potential)

@njit
def u(q, potential, eps=1e-4):
    '''
    yukawa (hopefully)
    '''
    if q < eps:
        return u_0(potential)
    else:
        return (2.0 * potential) / (q ** 2 + 1) 

@njit
def Iu(qvals, deltaq, potential):
    '''
    This is the integral from 0 to infinity of the absolute value of u(q) dq,
    due to the absolute value, this has no nice analytical expression. Therefore,
    an approximation with sum and infinity going to the max q value multiplied by
    deltaq
    '''
    I_sum = 0.0
    for i in range(qvals.shape[0]):
        q = qvals[i]
        I_sum += abs(u(q, potential)) * deltaq
    return I_sum

@njit
def If(qvals, deltaq, ZA, H, potential, I_u):
    '''
    This is the integral from 0 to infinity of the absolute value of f(q) dq,
    due to the absolute value.
    '''
    return (HistogramSum(H) * I_u) / ZA

@njit
def Fq(q, ZA, qvals, deltaq, H, potential, I_u):
    '''
    q here can be q,qprime,qdprime
    '''
    bin_q = get_bin(q, deltaq)
    return (H[bin_q] / ZA) * I_u

@njit
def seedchi():
    chi = np.random.uniform(-1, 1)
    return chi

@njit
def IndexWeights(weights):
    total = 0.0
    for i in range(len(weights)):
        total += weights[i]
    if total <= 0:
        return -1

    r = np.random.random() * total
    c = 0.0

    for i in range(len(weights)):
        c += weights[i]
        if r <= c:
            return i
    return weights.shape[0] - 1

@njit
def UniBin(bin_index, deltaq):
    return (bin_index) * deltaq 


@njit
def Uweights(qvals, deltaq, potential):
    n = qvals.shape[0]
    w = np.empty(n, dtype=np.float64)

    for i in range(n):
        q = qvals[i]
        if q == 0:
            uq = u_0(potential)
        else:
            uq = u(q, potential)
        w[i] = abs(uq) * deltaq

    return w


@njit
def PXY(RXY):
    return min(1.0, RXY)

@njit
def usqrt(q, q1, chi):
    '''
    sqrt(q^2 +q1^2 - 2qq1chi)
    q := q or qprime
    q1 := qprime or qdprime
    '''
    return np.sqrt(q**2 + q1**2 - (2.0*q*q1*chi))

# ============================ SAMPLING ============================

@njit
def BuildCDF(weights, cdf_out):
    total = 0.0
    for i in range(weights.shape[0]):
        total += weights[i]
        cdf_out[i] = total 
    return total 

@njit
def SampleCDF(cdf, total):
    if total <= 0.0:
        return -1

    r = np.random.random() * total
    lo = 0
    hi = cdf.shape[0] - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if r <= cdf[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo

@njit
def SeedFromCDF(cdf, total, deltaq):
    i = SampleCDF(cdf, total)
    if i == -1:
        return np.random.random() * (cdf.shape[0] * deltaq)
    return UniBin(i, deltaq)

# ============================ A -> A ============================

@njit
def RAA():
    return 1.0

# ============================ A -> B, B -> A ============================

@njit
def RAB_fast(q, qprime, PAB, lam, If_frozen, potential, chi):
    numerator1 = 2.0 * abs(1.0 + lam) * If_frozen
    denominator1 = np.pi * PAB
    magnitude_term = abs(u(usqrt(q, qprime, chi), potential) / u(q, potential))
    return (numerator1 / denominator1) * magnitude_term

# ============================ A -> C, C -> A ============================

@njit
def RAC(q, qprime, PCA, PAC, lam, qvals, deltaq, potential, chi, I_u):
    numerator1 = 2.0 * abs(lam) * I_u * PCA
    denominator1 = np.pi * PAC
    magnitude_term = abs(u(usqrt(q, qprime, chi), potential) / u(q, potential))
    return (numerator1 / denominator1) * magnitude_term

@njit
def RCA(q, qprime, PCA, PAC, lam, qvals, deltaq, potential, chi, I_u):
    return 1.0 / RAC(q, qprime, PCA, PAC, lam, qvals, deltaq, potential, chi, I_u)

# ============================ C -> D, D -> C ============================

@njit
def RCD_fast(qprime, qdprime, PCD, If_frozen, potential, chi):
    numerator1 = 2.0 * If_frozen
    denominator1 = np.pi * PCD
    magnitude_term = abs(u(usqrt(qprime, qdprime, chi), potential) / u(qprime, potential))
    return (numerator1 / denominator1) * magnitude_term

# ============================ HISTOGRAM UPDATER ============================

@njit
def HistogramBuilder(Type, q, qprime, qdprime, chi, potential,
                     ZA_frozen, qvals, deltaq, H_frozen, H_measured, I_u, chip,lamb):

    bin_q = get_bin(q, deltaq)

    if Type == 0:
        sign = -np.sign(u(q, potential))
    elif Type == 1:
        sign = -np.sign(u(usqrt(q, qprime, chi), potential) *
                        Fq(qprime, ZA_frozen, qvals, deltaq, H_frozen, potential, I_u))
    elif Type == 2:
        sign = -np.sign(u(usqrt(q, qprime, chi), potential) * u(qprime, potential))
    elif Type == 3:
        sign = -np.sign(u(usqrt(q, qprime, chi), potential) *
                        u(usqrt(qprime, qdprime, chip), potential) *
                        Fq(qdprime, ZA_frozen, qvals, deltaq, H_frozen, potential, I_u))

    H_measured[bin_q] += sign * (np.sign(lamb))
    return H_measured

# ============================ SCATTERING APPROX ============================

@njit
def ScatteringApprox(ZA, potential, qvals, deltaq, H, I_u):
    multiplier = (2.0 * I_u) / (np.pi * ZA)
    running_total = 0.0
    u_0val = u_0(potential)
    for s in range(H.shape[0]):
        running_total += (u(qvals[s], potential) * H[s])
    term = multiplier * running_total
    return u_0val + term

# ============================ ALGORITHM ============================

@njit
def BLDMC(PAA, PAB, PAC, PCA, PCD, PBA, PDC, N, M, q0, deltaq,
          potential, lam, H_approx, ZA_approx): 
    '''
    DIAGRAM A -> TYPE 0
    DIAGRAM B -> TYPE 1
    DIAGRAM C -> TYPE 2
    DIAGRAM D -> TYPE 3

    H_approx is the histogram from the previous trial
    ZA_approx is the sum from the that previous trial
    '''
    Type = 0
    Nbins = int(q0 / deltaq)
    qvals = np.arange(Nbins) * deltaq

    u_weights = Uweights(qvals, deltaq, potential)
    cdf_u = np.empty(Nbins, dtype=np.float64) # This is updated within the function below, however it is NOT returned, just updated.
    total_u = BuildCDF(u_weights, cdf_u)

    H_measured = np.zeros(Nbins)
    H_frozen = np.zeros(Nbins)

    I_u = Iu(qvals, deltaq, potential)

    # initial q from |u|
    q = SeedFromCDF(cdf_u, total_u, deltaq)

    qprime = 0.0
    qdprime = 0.0
    chi = 0.0
    chip = 0.0

    ZA_frozen = 0
    DiagramAsum = 0
    DiagramBsum = 0
    DiagramCsum = 0
    DiagramDsum = 0

    scattering_length_array = []

    absH = np.empty(Nbins, dtype=np.float64)
    cdf_h = np.empty(Nbins, dtype=np.float64)
    total_h = 0.0
    If_frozen = 0.0

    for i in range(M):
        H_measured[:] = 0.0
        DiagramAsum = 0
        DiagramBsum = 0
        DiagramCsum = 0
        DiagramDsum = 0

        if i == 0 and H_approx is not None:
            H_frozen = H_approx
            ZA_frozen = ZA_approx

        for s in range(Nbins):
            absH[s] = abs(H_frozen[s])
        total_h = BuildCDF(absH, cdf_h)

        if ZA_frozen > 0:
            If_frozen = (total_h * I_u) / ZA_frozen
        else:
            If_frozen = 0.0

        for _ in range(N):
            r = np.random.random()

            if Type == 0:
                if r < PAA:
                    q = SeedFromCDF(cdf_u, total_u, deltaq)
                    r2 = np.random.random()
                    if r2 < RAA():
                        Type = 0

                elif r < (PAA + PAB):
                    chi = seedchi()
                    qprime = SeedFromCDF(cdf_h, total_h, deltaq)
                    r2 = np.random.random()
                    if r2 < PXY(RAB_fast(q, qprime, PAB, lam, If_frozen, potential, chi)):
                        Type = 1

                else:  # r < (PAA + PAB + PAC)
                    chi = seedchi()
                    qprime = SeedFromCDF(cdf_u, total_u, deltaq)
                    r2 = np.random.random()
                    if r2 < PXY(RAC(q, qprime, PCA, PAC, lam, qvals, deltaq, potential, chi, I_u)):
                        Type = 2

            elif Type == 1:
                if r < PBA:
                    r2 = np.random.random()

                    if r2 < PXY(1.0 / RAB_fast(q, qprime, PAB, lam, If_frozen, potential, chi)):
                        Type = 0

            elif Type == 2:
                if r < PCA:
                    r2 = np.random.random()
                    if r2 < PXY(RCA(q, qprime, PCA, PAC, lam, qvals, deltaq, potential, chi, I_u)):
                        Type = 0
                else:  # r < (PCA + PCD)
                    chip = seedchi()
                    qdprime = SeedFromCDF(cdf_h, total_h, deltaq)
                    r2 = np.random.random()
                    if r2 < PXY(RCD_fast(qprime, qdprime, PCD, If_frozen, potential, chip)):
                        Type = 3

            elif Type == 3:
                if r < PDC:
                    r2 = np.random.random()

                    if r2 < PXY(1.0 / RCD_fast(qprime, qdprime, PCD, If_frozen, potential, chip)):
                        Type = 2

            if Type == 0:
                H_measured = HistogramBuilder(0, q, qprime, qdprime, chi,
                                              potential, ZA_frozen, qvals, deltaq,
                                              H_frozen, H_measured, I_u, chip,lam)
                DiagramAsum += 1
            elif Type == 1:
                H_measured = HistogramBuilder(1, q, qprime, qdprime, chi,
                                              potential, ZA_frozen, qvals, deltaq,
                                              H_frozen, H_measured, I_u, chip,lam)
                DiagramBsum += 1
            elif Type == 2:
                H_measured = HistogramBuilder(2, q, qprime, qdprime, chi,
                                              potential, ZA_frozen, qvals, deltaq,
                                              H_frozen, H_measured, I_u, chip,lam)
                DiagramCsum += 1
            elif Type == 3:
                H_measured = HistogramBuilder(3, q, qprime, qdprime, chi,
                                              potential, ZA_frozen, qvals, deltaq,
                                              H_frozen, H_measured, I_u, chip,lam)
                DiagramDsum += 1

        cumulativeZA = True
        if cumulativeZA:
            if DiagramAsum > 0:
                ZA_frozen = DiagramAsum
                H_frozen = H_measured.copy()
            else:
                print('0 skipped')
                continue
        else: # Doesn't work
            ZA_frozen += DiagramAsum
            H_frozen += H_measured

        approx = ScatteringApprox(ZA_frozen, potential, qvals, deltaq, H_frozen, I_u)
        scattering_length_array.append(approx)

        #print(approx)
        #print(f'DiagramA SUM: {DiagramAsum}')
        #print(f'DiagramB SUM: {DiagramBsum}')  
        #print(f'DiagramC SUM: {DiagramCsum}')
        #print(f'DiagramD SUM: {DiagramDsum}')
        #print()

    return approx, H_frozen, ZA_frozen

def a_analytical(potential):
    '''
    The potential here is defined V = -V_0, the input parameter "potential" wants  V_0, so for -5, input 5
    '''
    kappa = np.sqrt(2 * potential)
    return (1 - np.tan(kappa) / (kappa))

# ============================ RUN ============================

potlist = np.linspace(-0.1, -4.5, 600)
num = []
ann = []

H_approx = None
ZA_approx = 0

for i in potlist:
    approx, H_frozen, ZA_frozen = BLDMC(
        PAA=0.2, PAB=0.4, PAC=0.4,
        PCA=0.5, PCD=0.5,
        PBA=1, PDC=1,
        N=100000000, M=10,
        q0=10, deltaq=10/300,
        potential=i, lam=1,
        H_approx=H_approx, ZA_approx=ZA_approx
    )
    H_approx = H_frozen
    ZA_approx = ZA_frozen
    num.append(approx)
    print(approx)


plt.plot(potlist, num,color = 'red', label='Numerical')
phaseshifts, slength, gvals = ScatterLength(l=0 ,E=1e-3,rmax=400, h=0.001,mu=1,Gmin=0.1,Gmax = 4.5, N = 20000)
gvals = np.array(gvals)
gvals = -gvals
plt.plot(gvals,slength,'--k', label = 'Numerov')

plt.xlabel('Coupling Strength $g$')
plt.ylabel('Scattering Length')
plt.grid()
plt.ylim(-20, 20)

plt.legend()
plt.show()
