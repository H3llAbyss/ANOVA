'''

How to use my laboratornaya 2:

All of the steps are coded in ANOVA function. It does not have any outputs but print in console

ANOVA(data -> table of experiments
      r -> critical values of Duncan`s multiple range test (taken from table, hence is input)
      reverse -> should you reverse table (swap rows and tables)
      )
'''





import numpy as np
from scipy.stats import f,t


def ANOVA(data,r,reverse):


    n = len(data[0])
    k = len(data)

    if reverse:
        new_data = [[0 for i in range(k)] for j in range(n)]
        for i in range(n):
            for j in range(k):
                new_data[i][j] = data[j][i]
        data = new_data

    n = len(data[0])
    k = len(data)

    N = n * k
    x = data
    alpha = 0.05

    # 1
    print("Возможность воспроизводимости опытов (однородность дисперсий по критерию Кохрена):")

    s_sq = [0 for i in range(k)]

    for i in range(k):
        sum = 0
        for j in range(n):
            sum += pow((x[i][j] - np.mean(x[i])), 2)
        s_sq[i] = 1/(n-1) * sum

    F_H = max(s_sq)/min(s_sq)
    # print(F_H)
    F_crit = f.ppf(1-alpha, k-1, n*k-k)
    print("H0 is correct" if F_H < F_crit else "H0 is incorrect. Значит, подобные опыты не воспроизводимы!!!")

    # 2
    print("\nЭффекты фактора на всех уровнях отсутствуют:")

    s_sq_ouu = 0
    s_sq_Fi = 0
    F_H = 0
    sum_i = 0
    for i in range(k):
        sum_j = 0
        for j in range(n):
            sum_j += (x[i][j]-np.mean(x[i])) ** 2
        sum_i+=sum_j

    s_sq_ouu = sum_i
    #print(f"s_sq_ouu = {s_sq_ouu}")

    sum = 0
    for i in range(k):
        sum += (np.mean(x[i]) - np.mean(x)) ** 2
    s_sq_Fi = n * sum
    #print(f"s_sq_Fi = {s_sq_Fi}")

    F_H = (s_sq_Fi / (k-1)) / (s_sq_ouu / (N-k))

    F_crit = f.ppf(1-alpha, k - 1, N - k)
    #print(F_H, F_crit)

    print("H0 is correct" if F_H < F_crit else "H0 is incorrect. Значит эффект фактора присутствует как минимум на одном"
                                               " уровне")

    # 3
    print("\nОценить параметры модели, найти доверительные интервалы с доверительной вероятностью 95%:")

    t_quantile = t.ppf(1 - alpha / 2, N-k)
    mu = np.mean(x)
    tau_hat = [np.mean(x[i]) - np.mean(x) for i in range(k)]

    D_ouu = s_sq_ouu/(N-k)

    print(f"mu_hat ={np.mean(x)}")
    for i in range(k):
        print(f"tau_hat for A{i+1} = {tau_hat[i]}")

    # finding interval for M_i
    M_lower = [np.mean(x[i]) - t_quantile * np.sqrt(D_ouu/n) for i in range(k)]
    M_upper = [np.mean(x[i]) + t_quantile * np.sqrt(D_ouu/n) for i in range(k)]
    for i in range(k):
        print(f"For A{i+1} M_i bounds are between {M_lower[i]} and {M_upper[i]}")


    # finding interval for tau_i
    tau_lower = [tau_hat[i] - t_quantile * np.sqrt(D_ouu/n) for i in range(k)]
    tau_upper = [tau_hat[i] + t_quantile * np.sqrt(D_ouu/n) for i in range(k)]
    for i in range(k):
        print(f"For A{i+1} tau_i bounds are between {tau_lower[i]} and {tau_upper[i]}")

    # 4
    print("\nПостроить ряд предпочтений уровней фактора с использованием множественного критерия размахов Дункана:")
    x_bar = [np.mean(x[i]) for i in range(k)]
    x_bar_sorted = x_bar.copy()
    x_bar_sorted.sort()
    #print(x_bar_sorted)

    S = [np.sqrt((s_sq_ouu/(N-k))/k) for i in range(k)]

    df = N-k

    # r_i - input

    R = [r[i]*S[i] for i in range(1,k)]
    R.insert(0,0)

    crit = True
    for i in range(k-1,0 ,-1):
        if x_bar_sorted[i] - max(x_bar_sorted) > R[i]:
            crit = False
            break
    print("Разнца незначима" if crit else "Разница значима")


    print(x_bar)





# row i - number of test-1 (i == 0 means test 1)
# column j - different solutions (A1-A4)
#
#         A1     A2    A3    A4
data1 = [[13.2, 14.2, 16.7, 13.1],
         [14.1, 14.1, 18.1, 14.1],
         [13.3, 15.1, 16.2, 13.4],
         [13.1, 15.0, 18.4, 13.3],
         [13.0, 15.1, 15.5, 15.1],
         [15.2, 16.0, 16.4, 12.6],
         [12.3, 15.3, 17.0, 13.1],
         [13.5, 15.5, 18.0, 12.4],
         [15.1, 13.3, 17.1, 15.3],
         [14.0, 15.1, 17.7, 14.2]]

critical1 = [0, 3.846,4.011,4.121]

#ANOVA(data1,critical1,1)

data2 = [[2.294, 2.293, 2.289, 2.295, 2.298, 2.291, 2.300, 2.282],
         [2.309, 2.311, 2.310, 2.298, 2.287, 2.301, 2.307, 2.301],
         [2.301, 2.309, 2.304, 2.307, 2.300, 2.299, 2.310, 2.311]]

critical2 = [0, 2.941,3.088] #N-k = 8*3-3 = 21, i = 2,3

#ANOVA(data2,critical2,0)

data3 = [[10, 8 , 12, 12, 24, 19],
         [11, 10, 17, 15, 16, 18],
         [9 , 16, 14, 16, 22, 27],
         [13, 13, 9 , 16, 18, 25],
         [7 , 12, 16, 19, 20, 24]]
critical3 = [0,2.919,3.066,3.160,3.226,3.276] #N-k = 5*6-6 = 24, i = 2,3,4,5

ANOVA(data3,critical3,1)