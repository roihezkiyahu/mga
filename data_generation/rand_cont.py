import numpy as np
import scipy.stats as stats
import scipy.special


def generate_continuous_function(x_min, x_max, rate=1.0, apply_noise=False):
    x = np.linspace(x_min, x_max, 1000)
    amp_sin, freq_sin, phase_sin = np.random.uniform(0, 5) * rate, np.random.uniform(0, 2) * rate, np.random.uniform(0, 2 * np.pi)
    amp_cos, freq_cos, phase_cos = np.random.uniform(0, 5) * rate, np.random.uniform(0, 2) * rate, np.random.uniform(0, 2 * np.pi)
    sine_component, cosine_component = amp_sin * np.sin(freq_sin * x + phase_sin), amp_cos * np.cos(freq_cos * x + phase_cos)
    noise_component = np.random.uniform(0, 0.2) * rate * np.random.randn(len(x)) if apply_noise else np.zeros_like(x)
    trend_component = np.random.uniform(-1, 1) * rate * x + np.random.uniform(-5, 5)
    return x, sine_component + cosine_component + noise_component + trend_component


def gaussian_function(x_min, x_max, rate=1.0):
    x = np.linspace(x_min, x_max, 1000)
    mean, std = np.random.uniform(x_min, x_max), np.random.uniform(0.5, 5) * rate
    return x, stats.norm.pdf(x, mean, std)


def beta_function(x_min, x_max, rate=1.0):
    x = np.linspace(x_min, x_max, 1000)
    a, b = np.random.uniform(0.5, 5) * rate, np.random.uniform(0.5, 5) * rate
    return x, stats.beta.pdf(x, a, b)


def poisson_function(x_min, x_max, rate=1.0):
    lam = np.random.uniform(1, 20) * rate
    x = np.linspace(x_min, x_max, 1000)
    return x, (np.exp(-lam) * lam**x) / scipy.special.gamma(x + 1)


def gamma_function(x_min, x_max, rate=1.0):
    x = np.linspace(x_min, x_max, 1000)
    shape, scale = np.random.uniform(0.5, 5) * rate, np.random.uniform(0.5, 2) * rate
    return x, stats.gamma.pdf(x, shape, scale=scale)


def broken_distribution(x_min, x_max, rate=1.0):
    x_mid = np.random.uniform(x_min, x_max)
    x1, y1 = random_continuous_function(x_min, x_mid, rate)
    x2, y2 = random_continuous_function(x_mid, x_max, rate)
    x = np.linspace(x_min, x_max, 1000)
    y = np.concatenate([y1[:len(y1)//2], y2[len(y2)//2:]])
    return x, y


def exponential_function(x_min, x_max, rate=1.0):
    a, b = np.random.uniform(0, 5) * rate, np.random.uniform(-2, 2) * rate
    x = np.linspace(x_min, x_max, 1000)
    return x, a * np.exp(b * x)


def logistic_function(x_min, x_max, rate=1.0):
    L, k, x0 = rate, np.random.uniform(-2, 2), np.random.uniform(x_min, x_max)
    x = np.linspace(x_min, x_max, 1000)
    return x, L / (1 + np.exp(-k * (x - x0)))


def tanh_function(x_min, x_max, rate=1.0):
    x = np.linspace(x_min, x_max, 1000)
    return x, np.tanh(rate * x)


def logarithm_function(x_min, x_max, rate=1.0):
    a, base = rate, np.random.uniform(0.5, 2)
    x = np.linspace(x_min, x_max, 1000)
    return x, a * np.log(x + 1) / np.log(base)


def polynomial_function(x_min, x_max, rate=1.0):
    coeffs = [np.random.uniform(-rate, rate) for _ in range(np.random.randint(1, 5))]
    x = np.linspace(x_min, x_max, 1000)
    return x, np.polyval(coeffs, x)


def random_continuous_function(x_min, x_max, rate=1.0, apply_noise=True, choice=None):
    if not choice:
        choice = np.random.choice(
            ['custom', 'gaussian', 'beta', 'poisson', 'gamma', 'broken', 'exponential', 'logistic', 'tanh', 'logarithm', 'polynomial'],
            p=[0.3, 0.05, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.05, 0.1, 0.05]
        )
    functions = {
        'custom': generate_continuous_function,
        'gaussian': gaussian_function,
        'beta': beta_function,
        'poisson': poisson_function,
        'gamma': gamma_function,
        'broken': broken_distribution,
        'exponential': exponential_function,
        'logistic': logistic_function,
        'tanh': tanh_function,
        'logarithm': logarithm_function,
        'polynomial': polynomial_function
    }
    return functions[choice](x_min, x_max, rate, apply_noise) if choice == 'custom' else functions[choice](x_min, x_max, rate)


def random_operation_on_functions(x_min, x_max, rate=1.0):
    distributions = ['custom', 'gaussian', 'beta', 'poisson', 'gamma', 'broken', 'exponential', 'logistic',
                     'tanh', 'logarithm', 'polynomial']
    chosen_dists = np.random.choice(distributions, size=2, replace=False)
    x1, y1 = random_continuous_function(x_min, x_max, rate, choice=chosen_dists[0])
    x2, y2 = random_continuous_function(x_min, x_max, rate, choice=chosen_dists[1])
    x = x1
    operation = np.random.choice(['add', 'multiply', 'subtract', 'divide'])
    operations = {
        'add': lambda a, b: a + b,
        'multiply': lambda a, b: a * b,
        'subtract': lambda a, b: a - b,
        'divide': lambda a, b: a / (b + 1e-5)
    }
    return x, operations[operation](y1, y2)


def random_operation_n_times(x_min, x_max, rate=1.0, n_operations=10):
    distributions = ['custom']

    # Initial function
    x, y_result = random_continuous_function(x_min, x_max, rate, choice=np.random.choice(distributions),
                                             apply_noise=False)

    operations = {
        'add': lambda a, b: a + b,
        'subtract': lambda a, b: a - b
    }

    for _ in range(n_operations):
        _, y_new = random_continuous_function(x_min, x_max, rate, choice=np.random.choice(distributions),
                                              apply_noise=False)
        operation = np.random.choice(list(operations.keys()))
        y_result = operations[operation](y_result, y_new)

    return x, y_result