import matplotlib.pyplot as plt


def plot(errs):
    for k, v in errs.items():
        for a, e in v.items():
            if a in ['exact']:
                continue
            plt.semilogy(e.mean(1), label=a)
        plt.legend(loc='best')
        plt.title(k)
        plt.xlabel('n')
        plt.ylabel(r'$\frac{\|K - \hat{K}\|}{\|K\|}$')
        plt.show()
