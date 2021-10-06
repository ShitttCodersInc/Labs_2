from statistics import mean, variance
from scipy import stats
import numpy as np
n = 50

X = np.array([i+1 for i in range(50)])
Y = np.array([3.3728, 2.4136, 1.2693, 0.3779, -0.0775, -1.8075, -0.8770, -2.6304, -5.0338, -5.5773,
 -6.0536, -7.8564, -8.2299, -8.4545, -10.0819, -11.0336, -12.5008, -13.2778, -14.6758, -14.8179,
 -16.2261, -16.6083, -17.3814, -18.4614, -20.0174, -21.2520, -22.6120, -22.9407, -23.5203, -25.1705,
 -26.0867, -26.6918, -27.5680, -29.7085, -29.9000, -30.9021, -32.0756, -32.5536, -34.1821, -35.4131,
 -35.8620, -36.7701, -38.1088, -38.6019, -40.7590, -41.5375, -43.5361, -42.7393, -44.4955, -45.1266])

alpha1 = 0.01
alpha2 = 0.05

x_avg, y_avg = mean(X), mean(Y)
S_x, S_y = variance(X), variance(Y)

r_xy = (sum((X[k] - x_avg) * (Y[k] - y_avg) for k in range(n)) / (n - 1))

#------------------------------------
# Коэффициенты уравления регрессии

get_b1 = lambda r_xy, S_y, S_x: r_xy * S_y / S_x
get_b0 = lambda y_agv, b1, x_agv: y_agv - b1 * x_agv

b1=get_b1(r_xy=r_xy, S_y=S_y, S_x=S_x)
b0=get_b0(y_agv=y_avg, b1=b1, x_agv=x_avg)
#--------------------------------
# Проверка значимости коэффициентов

S_y_star = ((1/(n-2)) * (sum(Y[k] - b0 - b1 * X[k] for k in range(n)))**2)**0.5

get_S_b0 = lambda S_y_star, n, x_agv, S_x: S_y * (1/n + x_avg ** 2 / ((n-1) * S_x ** 2)) ** 0.5
get_S_b1 = lambda S_y_star, n, X: ((( S_y_star**2) * n) /(n * (sum(X[k] ** 2 for k in range(n))) - sum(X[k] for k in range(n)) ** 2 )) ** 0.5

S_b0 = get_S_b0(S_y_star=S_y_star, n=n, x_agv=x_avg, S_x=S_x)
S_b1 = get_S_b1(S_y_star=S_y_star, n=n, X=X)

delta_b0 = stats.t.ppf(alpha1, 48) * S_b0
delta_b1 = stats.t.ppf(alpha2, 48) * S_b1

print('Коэффициент b0 значим для alpha = 0.01') if abs(b0) > abs(delta_b0) else print('Коэффициент b0 не значим для alpha = 0.01')
print('Коэффициент b0 значим для alpha = 0.05') if abs(b0) > abs(delta_b1) else print('Коэффициент b0 не значим для alpha = 0.05')
print('Коэффициент b1 значим для alpha = 0.01') if abs(b1) > abs(delta_b0) else print('Коэффициент b1 не значим для alpha = 0.01')
print('Коэффициент b1 значим для alpha = 0.05') if abs(b1) > abs(delta_b1) else print('Коэффициент b1 не значим для alpha = 0.05')
#-------------------------------
# Проверка адекватности модели

get_f = lambda x, b0, b1: b0 + b1 * x
f = get_f(x=X, b0=b0, b1=b1)

S_y_out = ((1/(n-1)) * sum(Y[k] - y_avg for k in range(n)) ** 2) ** 0.5
S_y_ost = ((1/(n-3)) * sum(Y[k] - f[k] for k in range(n)) ** 2) ** 0.5

# Экспериментальное значение F-критерия

F = S_y_out / S_y_ost

print('Уравнение регрессии адекватно для alpha = 0.01') if F > stats.f.ppf(alpha1, n-1, n-2) else print('Уравнение регрессии адекватно для alpha = 0.01')
print('Уравнение регрессии адекватно для alpha = 0.05') if F > stats.f.ppf(alpha2, n-1, n-2) else print('Уравнение регрессии адекватно для alpha = 0.05')
