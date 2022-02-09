import torch
import torch.nn as nn
import torch.nn.functional as F

class FFT(nn.Module):
    def __init__(self):
        super(FFT,self).__init__()

    def forward(self,x):
        x=torch.fft.fft2(x, dim=(-2, -1))
        x=torch.stack((x.real, x.imag), -1)
        return x

class IFFT(nn.Module):
    def __init__(self):
        super(IFFT, self).__init__()

    def forward(self,x):
        res = torch.fft.ifft2(torch.complex(x[..., 0], x[..., 1]), dim=(-2, -1)).real
        return res

'''
    Fourier convolutional layer 2.0
'''
class FourierConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias = True):
        super(FourierConv2D, self).__init__()

        self.kernel = nn.Parameter(data = torch.Tensor(out_channels, in_channels, kernel_size, kernel_size), requires_grad = True)
        self.kernel.data.uniform_(-0.1, 0.1)

        if bias:
            self.bias = nn.Parameter(data = torch.Tensor(out_channels), requires_grad = True)
            self.bias.data.zero_()
        else:
            self.bias = None

        self.out_channels, self.in_channels, self.kernel_size = out_channels, in_channels, kernel_size

    def complex_element_wise_product(self, x, y):
        xr, xi = x[..., 0], x[..., 1]
        yr, yi = y[..., 0], y[..., 1]
        return torch.stack([xr * yr - xi * yi, xr * yi + xi * yr], dim = -1)

    def forward(self, img):
        bs, _, h, w,_ = img.size()

        new_h, new_w = h, w
        pad_h, pad_w = (new_h - h) // 2, (new_w - w) // 2

        new_img = img#F.pad(img, (0,0,pad_w, pad_w, pad_h, pad_h), mode = 'constant', value = 0)

        new_kernel = torch.zeros([self.out_channels, self.in_channels, new_h, new_w]).to(self.kernel.device)
        k_center = self.kernel_size // 2
        ky, kx = torch.meshgrid(torch.arange(self.kernel_size), torch.arange(self.kernel_size))

        kny = (ky.flip(0) - k_center) % new_h
        knx = (kx.flip(1) - k_center) % new_w
        new_kernel[..., kny, knx] = self.kernel[..., ky, kx]

        #new_img_fft_complex = torch.fft.fft2(new_img, dim=(-2, -1))
        #new_img_fft=torch.stack((new_img_fft_complex.real, new_img_fft_complex.imag), -1)
        new_kernel_fft_complex = torch.fft.fft2(torch.transpose(new_kernel, 0, 1), dim=(-2, -1))
        new_kernel_fft=torch.stack((new_kernel_fft_complex.real, new_kernel_fft_complex.imag), -1)

        res_fft = torch.zeros(bs, self.out_channels, new_h, new_w, 2).to(img.device)

        for i in range(self.in_channels):
            res_fft = res_fft + self.complex_element_wise_product(new_img[:, i, ...].unsqueeze(1), new_kernel_fft[i, ...].unsqueeze(0))
        #res = torch.fft.ifft2(torch.complex(res_fft[..., 0], res_fft[..., 1]), dim=(-2, -1)).real

        if not self.bias==None:
            res_fft#[..., pad_h:-pad_h, pad_w:-pad_w,0]+=self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        return res_fft#[..., pad_h:-pad_h, pad_w:-pad_w,:]

'''
    Spectral pooling layer
'''
class SpectralPooling2d_kernel(nn.Module):
    def __init__(self, kernel_size):
        super(SpectralPooling2d_kernel, self).__init__()

        self.kernel_size = kernel_size

    def crop_spectrum(self, z, H, W):
        '''
            z: [bs, c, M, N, 2]
            Return: [bs, c, H, H, 2]
        '''
        M, N = z.size(-3), z.size(-2)
        return z[..., M//2-H//2:M//2+H//2, N//2-W//2:N//2+W//2, :]

    def pad_spectrum(self, z, M, N):
        '''
            z: [bs, c, H, W, 2]
            Return: [bs, c, M, N, 2]
        '''
        H, W = z.size(-3), z.size(-2)
        z_real, z_imag = z[..., 0], z[..., 1]
        pad = torch.nn.ZeroPad2d((N-W)//2, (N-W)//2, (M-H)//2, (M-H)//2)
        return torch.stack([pad(z_real), pad(z_imag)], dim = -1)

    def treat_corner_cases(self, freq_map):
        '''
            freq_map: [bs, c, M, N, 2]
        '''
        S = [(0, 0)]
        M, N = freq_map.size(-3), freq_map.size(-2)

        if M % 2 == 1:
            S.append((M // 2, 0))
        if N % 2 == 1:
            S.append((0, N // 2))
        if M % 2 == 1 and N % 2 == 1:
            S.append((M // 2, N // 2))

        for h, w in S:
            freq_map[..., h, w, 1].zero_()

        return freq_map, S

    def remove_redundancy(self, y):
        '''
            y: input gradient map [bs, c, M, N, 2]
        '''
        z, S = self.treat_corner_cases(y)
        I = []
        M, N = y.size(-3), y.size(-2)

        for m in range(M):
            for n in range(N // 2 + 1):
                if (m, n) not in S:
                    if (m, n) not in I:
                        z[..., m, n, :].mul_(2)
                        I.append((m, n))
                        I.append(((M - m) % M, (N - n) % N))
                    else:
                        z[..., m, n, :].zero_()
        
        return z

    def recover_map(self, y):
        z, S = self.treat_corner_cases(y)
        I = []
        M, N = y.size(-3), y.size(-2)

        for m in range(M):
            for n in range(N // 2 + 1):
                if (m, n) not in S:
                    if (m, n) not in I:
                        z[..., m, n, :].mul_(0.5)
                        z[..., (M-m)%M, (N-n)%N] = z[..., m, n, :]
                        I.append((m, n))
                        I.append(((M - m) % M, (N - n) % N))
                    else:
                        z[..., m, n, :].zero_()

        return z

    def forward(self, x):
        M, N = x.size(-3), x.size(-2)
        H, W = M // self.kernel_size, N // self.kernel_size

        #x_fft_complex = torch.fft.fft2(x, dim=(-2, -1))
        #x_fft=torch.stack((x_fft_complex.real, x_fft_complex.imag), -1)
        crop_x_fft = self.crop_spectrum(x, H, W)
        crop_x_fft, _ = self.treat_corner_cases(crop_x_fft)
        pool_x = crop_x_fft#torch.fft.ifft2(torch.complex(crop_x_fft[..., 0], crop_x_fft[..., 1]), dim=(-2, -1)).real
        return pool_x

    def backward(self, gRgx):
        H, W = gRgx.size(-3), gRgx.size(-2)
        M, N = H * self.kernel_size, W * self.kernel_size

        #z_complex = torch.fft.fft2(gRgx, dim=(-2, -1))
        #z = torch.stack((z_complex.real, z_complex.imag), -1)
        z = gRgx
        z = self.remove_redundancy(z)
        z = self.pad_spectrum(z, M, N)
        z = self.recover_map(z)
        #gRx = torch.fft.ifft2(torch.complex(z[..., 0], z[..., 1]), dim=(-2, -1)).real
        gRx = z

        return gRx

class SpectralPooling2d(nn.Module):
    def __init__(self, kernel_size):
        super(SpectralPooling2d, self).__init__()

        self.kernel_size = kernel_size
        self.kernel=SpectralPooling2d_kernel(self.kernel_size)

    def forward(self,x):
        x = torch.fft.fftshift(torch.complex(x[..., 0], x[..., 1]), dim=(-2, -1))
        x = torch.stack((x.real, x.imag), -1)
        x= self.kernel(x)
        return x

class SpectralBN(nn.Module):
    def __init__(self,**kwargs):
        super(SpectralBN, self).__init__()
        self.BNreal=nn.BatchNorm2d(**kwargs)
        self.BNimag = nn.BatchNorm2d(**kwargs)

    def forward(self,x):
        return torch.stack((self.BNreal(x[...,0]),self.BNimag(x[...,1])),dim=-1)

if __name__ == '__main__':
    import time

    fc = FourierConv2D(1, 3, 3)
    # fc = nn.Conv2d(1, 3, 3)
    for param in fc.parameters():
        print(param)

    # with SummaryWriter('./log') as w:
    #     w.add_graph(fc, input_to_model = x)

    optim = torch.optim.Adam(fc.parameters(), lr = 0.1)

    for _ in range(6):
        x = torch.randn([5, 1, 6, 6], requires_grad = True)
        FFTlayer=FFT()
        x=FFTlayer(x)
        optim.zero_grad()
        start = time.time()
        y = fc(x)
        end = time.time()
        loss = torch.sum(y)
        loss.backward()
        optim.step()
        print(end - start)

    for param in fc.parameters():
        print(param)