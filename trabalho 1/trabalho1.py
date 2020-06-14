#%% importing modules
import numpy as np
import matplotlib.pyplot as p

#%% part 1: defining example signals
# gaussian parameters
A = 3           # amplitude
sigma = 10      # time width

# defining time array
N = int(1.2e4)  # size of the computational vector
                # if N is large the vector seems continuous

t_step = sigma / 100           # time step
t_i = -t_step * (N - 1) / 2    # time window start
t_f = -t_i                      # time window end
t = np.linspace(t_i, t_f, N)    # time array

# defining gaussian function
x = A * np.exp(-(t / sigma)**2)

# defining frequency array in Hertz
f_step = 1 / (50 * sigma)      # frequency step
f_i = -f_step * (N - 1) / 2    # initial frequency
f_f = -f_i                      # ending frequency
f = np.linspace(f_i, f_f, N)    # frequency array

# defining FT of x
X = A * sigma * np.sqrt(np.pi) * np.exp(-(np.pi * sigma * f)**2)

print(t[0], t[-1], x[0], x[-1])
print(f[0], f[-1], X[0], X[-1])

# plotng the results
fig = p.figure()
p.style.use('seaborn-dark')
fig.subplots_adjust(hspace=0.6)

ax1 = fig.add_subplot(211)
# p.subplot(211)
ax1.plot(t, x, 'r')
ax1.set_xlabel('tempo [s]')
ax1.set_ylabel('x(t)')
ax1.set_title('Função Gaussiana definida analiticamente')
p.grid()

ax2 = fig.add_subplot(212)
# p.subplot(212)
ax2.plot(f, X, 'b')
ax2.set_xlabel('frequência [Hz]')
ax2.set_ylabel('X(f)')
ax2.set_title('TF analítica de x(t)')
p.grid()

p.show()

'''
Questions:
1.1) Sampling period: T_s = sigma / 100 = 0.1 s. 
     Sampling rate: f_s = 1 / T_s = 10 Hz
1.3) f_0 = 10/10 = 1, where X(1) = 0.
     The greatest value for X(f) is when f = 0, X(0) = 53.121
14) Yes. The sample rate must be at least twice the maximum frequency,
     Which in this case is 1 Hz. As 10 > 2 * 1, the sampling period
     is enough to sample the function without losing information.
'''

#%% part 2: Calculating DTFT of x[n]

# creating an array of non-dimensional frequencies
v_i = -1.2
v_f = -v_i
v_step = (v_f - v_i) / (N - 1)
v = np.arange(v_i, v_f + v_step, v_step)

# calculating the DTFT of x[n] (defined on part 1)



# %%
