from astropy.io import fits
import math
import json
import lsp
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    with fits.open("ccd.fits.gz") as fits_file:
        data = fits_file[0].data.astype(np.int16)
    u0 = np.mean(data[0, 0, ...])
    x = np.mean(data[:, 0, ...],
                axis=(1, 2)) - u0
    sigma_Delta_x = np.var(data[:, 1, ...] - data[:, 0, ...],
                           axis=(1, 2))
    print(sigma_Delta_x)
    A = np.column_stack((x, np.ones_like(x)))
    xi, cost, var = lsp.lstsq_ne(A, sigma_Delta_x)

    k, b = xi[0], xi[1]
    plt.plot(x, k * x + b, label="линейная зависимость", color='brown')
    plt.scatter(x, sigma_Delta_x, label="зависимость из файла ccd.fits", color='grey')
    plt.xlabel("x")
    plt.ylabel("sigma_Delta_x**2")
    plt.legend()
    plt.savefig('ccd.png')
    plt.show()

    gain = 2 / k
    sigma_ron = math.sqrt(2 * b / k ** 2)
    data = {
        "ron": round(sigma_ron, 2),
        "gain": round(gain, 3)
    }

    with open('ccd.json', 'w') as f:
        json.dump(data, f)
