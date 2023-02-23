from lsp import lstsq
import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt

if __name__ == "__main__":
    n, m = 500, 20
    A = np.random.normal(size=(n, m))
    x = np.random.normal(size=(m,))
    sigma, imp = 0.01, 10000
    b_ten_tho = np.random.multivariate_normal(
        mean=np.dot(A, x),
        cov=np.eye(n) * sigma ** 2,
        size=imp)
    d = [lstsq(A, b)[1] for b in b_ten_tho]

    plt.hist(d, density=True, bins=50, alpha=0.6, label='Гистограмма величины невязки')
    df = n - m
    x = np.linspace(chi2.ppf(0.01, df),
                    chi2.ppf(0.99, df), 100)
    plt.plot(x * (sigma ** 2), chi2.pdf(x, df) / (sigma ** 2), 'r-', linewidth=5, alpha=0.6,
             label='теоретическое распределение')
    plt.xlabel("x")
    plt.ylabel("плотность распределения частот")
    plt.legend()
    plt.savefig('chi2.png')
    plt.show()
