import numpy as np

class delayLine:
    """
    Implementation of circular buffer for delay line by N frames.
    """
    def __init__(self, N, T = 1):
        self.N = N
        self.p = 0
        self.buffer = [0]*self.N
        self.T = T
    
    def IO(self, inp):
        """
        IO (input-output) implements circular buffer for delay line.
        inp: (float) value at time n
        out: (float) value at time n - N
        """
        if (self.N == 0):
            return inp
        else:
            out = self.buffer[self.p]
            self.buffer[self.p] = inp
            self.p = (self.p + 1) % self.N
        return out
    
    def reset(self):
        """
        resets buffer to all zeros
        """
        self.buffer = [0]*self.N
    
    def freqResponse(self):
        """
        Returns a function that implements the filter's frequency response
        """
        def f(omega):
            return np.exp(-1j*omega*self.N*self.T)
        return f

class digitalFilter:
    def __init__(self, a, b, T = 1):
        """
        H(z) = (b_0 + b_1 z^-1 + ... + b_M z^-M)/(1 + a_1 z^-1 + ... + a_N z^-N)
        a: vector of coefficients for denominator of transfer function a_, ..., a_N
        b: vector of coefficients for numerator of transfer function b_0, ..., b_M
        
        Note: this isn't the most efficient implementation because it makes use of a different buffer for each delay line
        """
        self.a = np.array(a)
        self.b = np.array(b)
        self.T = T
        
        self.Na = np.where(self.a != 0)[0]
        self.Nb = np.where(self.b != 0)[0]
        
        self.anz = self.a[self.Na]
        self.bnz = self.b[self.Nb]
        
        #Only create buffers for delays with nonzero magnitudes
        self.buffers_a = [delayLine(N, T = self.T) for N in self.Na]
        self.buffers_b = [delayLine(N, T = self.T) for N in self.Nb]
        
        self.out_prev = 0
        
    def IO(self, inp):
        """
        Gives the filter's next output given the current input
        """
        out = 0
        for i in range(len(self.Nb)):
            out += self.bnz[i] * self.buffers_b[i].IO(inp)
        if(len(self.Na) > 0):
            for j in range(len(self.Na)):
                out -= self.anz[j] * self.buffers_a[j].IO(self.out_prev)
        self.out_prev = out
        return out
    
    def reset(self):
        """
        resets all buffers
        """
        for i in range(len(self.Nb)):
            self.buffers_b[i].reset()
        for j in range(len(self.Na)):
            self.buffers_a[j].reset()
                
    def impulseResponse(self, N_t):
        """
        Gives the impulse response over N_t time frames. Returns (t, response)
        """
        self.reset()
        response = np.ndarray(N_t)
        t = np.linspace(0, self.T*(N_t - 1), N_t)
        response[0] = self.IO(1)
        for i in range(1, N_t):
            response[i] = self.IO(0)
        self.reset()
        return t, response
                
    def __freqResponse__(self):
        def f(omega):
            num = 0
            for i in range(len(self.Nb)):
                num += self.bnz[i]*self.buffers_b[i].freqResponse()(omega)
            denom = 1
            if(len(self.Nb) > 0):
                for j in range(len(self.Na)):
                    denom += np.exp(-1j*omega*self.T) * self.anz[j] * self.buffers_a[j].freqResponse()(omega)
            return num/denom
        return f
    
    def freqResponse(self, frac = 1, N = 10000):
        """
        Gives the magnitude frequency response over the frequency interval [0, frac*sampling rate]. N is the 
        number of sampled frequencies.
        returns (freq, reponse)
        """
        fr = self.__freqResponse__()
        fs = np.linspace(0, frac*2*np.pi*1/self.T, N)
        return fs/(2*np.pi), np.abs(fr(fs))