import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn import linear_model
import pandas as pd

# with open('ifelse.csv') as file:
#     df = pd.read_csv(file)
#     df['Speedup'] = (df['Scalar'] - df['Runtime Check']) / df['Scalar']
#     ax = df.plot(x="Branch Probability", y="Speedup")
#     ax.set_title('If-Else Runtime Check Performance Improvement Ratio (vs Scalar)')
#     ax.set_ylabel('Ratio (%)')
#     ax.set_xlabel('Branch Probability (%)')
#     plt.savefig('if-else-ratio.png')
#     # plt.show()
#
# with open('ifthen.csv') as file:
#     df = pd.read_csv(file)
#     df['Speedup'] = (df['Scalar'] - df['Runtime Check']) / df['Scalar']
#     ax = df.plot(x="Branch Probability", y="Speedup")
#     ax.set_title('If-Then Runtime Check Performance Improvement Ratio (vs Scalar)')
#     ax.set_ylabel('Ratio (%)')
#     ax.set_xlabel('Branch Probability (%)')
#     plt.savefig('if-then-ratio.png')
#     # plt.show()
#
# with open('ifthen.csv') as file:
#     df = pd.read_csv(file)
#     ax = df.plot(x="Branch Probability")
#     ax.set_title('If-Then Execution Time vs. Branch Probability')
#     ax.set_ylabel('Execution Time (sec)')
#     ax.set_xlabel('Branch Probability (%)')
#     plt.savefig('if-then-time.png')
#     # plt.show()
#
# with open('ifelse.csv') as file:
#     df = pd.read_csv(file)
#     ax = df.plot(x="Branch Probability")
#     ax.set_title('If-Else Execution Time vs. Branch Probability')
#     ax.set_ylabel('Execution Time (sec)')
#     ax.set_xlabel('Branch Probability (%)')
#     plt.savefig('if-else-time.png')
#     # plt.show()

def cost_4(p, m, n, s, t):
    return m*p**4 + n*(1-p)**4 + (1-(p**4+(1-p)**4))*(p*s+(1-p)*t)*4

def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x

    return (b_0, b_1)


def plot_regression_line(x, y, b):
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color="m",
                marker="o", s=30)

    # predicted response vector
    y_pred = b[0] + b[1] * x

    # plotting the regression line
    plt.plot(x, y_pred, color="g")

    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')

    # function to show plot
    plt.show()

with open('moldyn-scalar.csv') as scalar, open('moldyn-sse.csv') as sse, open('moldyn-avx.csv') as avx:
    df_scalar = pd.read_csv(scalar)
    df_sse = pd.read_csv(sse)
    df_avx = pd.read_csv(avx)
    df = pd.DataFrame({
        'Branch Probability': df_scalar['Branch Probability']*0.01,
        'Scalar': df_scalar['Time'],
        'Runtime Check VF=2': df_sse['Time'],
        'Runtime Check VF=4': df_avx['Time']
    })
    mlr = linear_model.LinearRegression()
    df['UF'] = (1- df['Branch Probability'])**4
    df['UT'] = df['Branch Probability']**4
    b_1 = 1.8468085521174318e-10
    # b_1 = 1E-20
    df['ST'] = df['Runtime Check VF=4'] - (1-df['UF'] - df['UT'])*(b_1 * df['Branch Probability'])*4
    print(df['Runtime Check VF=4'])
    print(df)
    # df['D'] = 1 - (df['UF'] + df['UT'])
    # mlr.fit(df[['UT', 'UF', 'D']], df['Runtime Check VF=4'])
    mlr.fit(df[['UT', 'UF']], df['ST'])
    df.drop('UF', axis=1, inplace=True)
    df.drop('UT', axis=1, inplace=True)
    # df.drop('D', axis=1, inplace=True)
    df.drop('ST', axis=1, inplace=True)

    # popt, pcov = curve_fit(cost_4, , df_avx['Time'])
    print(mlr.coef_)
    print(mlr.intercept_)
    # df['Fit'] = [mlr.intercept_ + mlr.coef_[0]*p**4 + mlr.coef_[1]*(1-p)**4 + (1-(p**4 + (1-p)**4))*mlr.coef_[2] for p in df['Branch Probability']]
    df['Fit'] = [mlr.intercept_ + mlr.coef_[0]*p**4 + mlr.coef_[1]*(1-p)**4 + (1-(p**4+(1-p)**4))*(b_1 * p)*4 for p in df['Branch Probability']]

    ax = df.plot(x='Branch Probability')
    ax.set_title('Moldyn Per-Iteration Execution Times vs. Branch Probability', y=1.08)
    ax.set_ylabel('Execution Time of One Loop Iteration (sec)')
    ax.set_xlabel('Branch Probability (%)')
    plt.savefig('moldyn.png')
    # plt.show()