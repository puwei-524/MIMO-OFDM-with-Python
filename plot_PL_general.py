import numpy as np
import matplotlib.pyplot as plt


def pl_free(fc, dist, Gt=1, Gr=1):
    """
    Free Space Path Loss Model

    Parameters:
    fc (float): Carrier frequency in Hz
    dist (float or np.ndarray): Distance between base station and mobile station in meters
    Gt (float): Transmitter gain (default is 1)
    Gr (float): Receiver gain (default is 1)

    Returns:
    float or np.ndarray: Path loss in dB
    """
    lamda = 3e8 / fc
    tmp = lamda / (4 * np.pi * dist)

    if Gt:
        tmp = tmp * np.sqrt(Gt)
    if Gr:
        tmp = tmp * np.sqrt(Gr)

    PL = -20 * np.log10(tmp)
    return PL


def pl_logdist_or_norm(fc, d, d0, n, sigma=None):
    """
    Log-distance or Log-normal Shadowing Path Loss model

    Parameters:
    fc (float): Carrier frequency in Hz
    d (float or np.ndarray): Distance between base station and mobile station in meters
    d0 (float): Reference distance in meters
    n (float): Path loss exponent
    sigma (float, optional): Standard deviation of shadowing in dB

    Returns:
    float or np.ndarray: Path loss in dB
    """
    lamda = 3e8 / fc
    PL = -20 * np.log10(lamda / (4 * np.pi * d0)) + 10 * n * np.log10(d / d0)

    if sigma is not None:
        PL += sigma * np.random.randn(*np.shape(d))

    return PL


# 参数设置
fc = 1.5e9  # 载波频率1.5GHz
d0 = 100  # 参考距离
sigma = 3  # 标准差
distance = np.arange(1, 32, 2) ** 2  # 距离
Gt = [1, 1, 0.5]  # 发射天线增益
Gr = [1, 0.5, 0.5]  # 接受天线增益
Exp = [2, 3, 6]  # 路径损耗指数

# 计算路径损耗
y_Free = np.zeros((3, len(distance)))
y_logdist = np.zeros((3, len(distance)))
y_lognorm = np.zeros((3, len(distance)))

for k in range(3):
    y_Free[k, :] = pl_free(fc, distance, Gt[k], Gr[k])  # 自由空间的路径损耗
    y_logdist[k, :] = pl_logdist_or_norm(fc, distance, d0, Exp[k])  # 对数路径损耗模型
    y_lognorm[k, :] = pl_logdist_or_norm(fc, distance, d0, Exp[0], sigma)  # 对数正态阴影衰落模型

# 绘制自由空间路径损耗模型
plt.figure(1)
plt.semilogx(distance, y_Free[0, :], 'k-o', label='G_t=1, G_r=1')
plt.semilogx(distance, y_Free[1, :], 'b-^', label='G_t=1, G_r=0.5')
plt.semilogx(distance, y_Free[2, :], 'r-s', label='G_t=0.5, G_r=0.5')
plt.grid(True)
plt.axis([1, 1000, 40, 110])
plt.title(f'Free PL Models, f_c={fc / 1e6} MHz')
plt.xlabel('Distance [m]')
plt.ylabel('Path loss [dB]')
plt.legend()

# 绘制对数路径损耗模型
plt.figure(2)
plt.semilogx(distance, y_logdist[0, :], 'k-o', label='n=2')
plt.semilogx(distance, y_logdist[1, :], 'b-^', label='n=3')
plt.semilogx(distance, y_logdist[2, :], 'r-s', label='n=6')
plt.grid(True)
plt.axis([1, 1000, 40, 110])
plt.title(f'Log-distance PL model, f_c={fc / 1e6} MHz')
plt.xlabel('Distance [m]')
plt.ylabel('Path loss [dB]')
plt.legend()

# 绘制对数正态阴影路径损耗模型
plt.figure(3)
plt.semilogx(distance, y_lognorm[0, :], 'k-o', label='path 1')
plt.semilogx(distance, y_lognorm[1, :], 'b-^', label='path 2')
plt.semilogx(distance, y_lognorm[2, :], 'r-s', label='path 3')
plt.grid(True)
plt.axis([1, 1000, 40, 110])
plt.title(f'Log-normal PL model, f_c={fc / 1e6} MHz, σ={sigma} dB')
plt.xlabel('Distance [m]')
plt.ylabel('Path loss [dB]')
plt.legend()

plt.show()
