import numpy as np
from math import factorial
from itertools import permutations


def vec_sample_from_distribution(P,
                                 x0,
                                 nsamples,
                                 decorrelation_steps,
                                 xmax=float('inf')):
    vec_batch = 100
    x = np.random.rand(vec_batch, x0.shape[0])
    step = np.full(x.shape, xmax)
    Px = P(x)
    samples_obtained = 0
    samples = []
    steps = 0
    count_true = 0
    while samples_obtained < nsamples:
        new_x = x + step * (np.random.rand(step.shape[0], step.shape[1]) - 0.5)
        Pnew_x = P(new_x)
        a = np.random.rand()
        condition = np.vstack(
            (Pnew_x / Px >= a, np.all(np.abs(new_x) < xmax, axis=1)))
        condition = np.all(condition, axis=0)
        x = np.where(
            np.full((x.shape[0], x.shape[1]),
                    np.repeat(condition,
                              x0.shape[0]).reshape(x.shape[0], x.shape[1])),
            new_x, x)
        Px = np.where(condition, Pnew_x, Px)
        if steps % decorrelation_steps == 0:
            if samples_obtained == 0:
                samples = x
            else:
                samples = np.append(samples, x, axis=0)
            samples_obtained += 1 * vec_batch
        steps += 1

    return samples


def sample_from_distribution(P,
                             x0,
                             step,
                             nsamples,
                             decorrelation_steps,
                             xmax=float('inf')):
    x = x0
    Px = P(x)
    samples_obtained = 0
    samples = []
    steps = 0
    count_true = 0
    while samples_obtained < nsamples:
        new_x = x + 2 * step * (np.random.rand(step.shape[0]) - 0.5)
        Pnew_x = P(new_x)
        if ((Px == 0).all() or (np.random.rand() <= Pnew_x / Px).all()) and (
                np.abs(new_x) < xmax).all():
            count_true += 1
            x = new_x
            Px = Pnew_x
        if steps % decorrelation_steps == 0:
            samples.append(x)
            samples_obtained += 1
        steps += 1

    return np.array(samples)


def sample_uniform(a, nsamples):
    X = (np.random.rand(*nsamples) - 0.5) * (2 * a)
    x_uniform = (np.random.rand(*nsamples) - 0.5) * (2 * a)
    return x_uniform


def sample_mixed(P,
                 x0,
                 step,
                 a,
                 nsamples,
                 decorrelation_steps,
                 uniform_ratio=0.5):
    assert 0.0 <= uniform_ratio and uniform_ratio <= 1.0, 'uniform_ratio must be between zero and one!'
    x_uniform = sample_uniform(a, [int(uniform_ratio * nsamples), x0.shape[0]])
    x_P = vec_sample_from_distribution(P, x0,
                                       int((1.0 - uniform_ratio) * nsamples),
                                       decorrelation_steps, a)
    x = np.concatenate([x_uniform, x_P])
    np.random.shuffle(x)
    return x


if __name__ == '__main__':
    import numpy as np, matplotlib
    from matplotlib import pyplot as plt
    step = np.array([1.0])
    x0 = np.array([0.0])
    nsamples = 10000
    decorrelation_steps = 5
    n_particles = 2

    def P(x):
        return np.exp(-x**2) + 0.5 * np.exp(-5.0 * (x - 2.0)**2)

    samples = sample_from_distribution(P, x0, step, nsamples,
                                       decorrelation_steps)
    fig = plt.figure()
    plt.hist(samples, bins='auto')
    xmax = 5.0
    step = np.array([xmax, xmax])
    x0 = np.array([0.0, 0.0])
    nsamples = 10000
    decorrelation_steps = 5

    def P2(x):
        return np.exp(-x[0]**2 - x[1]**2) + 0.5 * np.exp(
            -5.0 * (x[0] - 2.0)**2) * np.exp(-2.0 * x[1]**2)

    samples = sample_from_distribution(P2, x0, step, nsamples,
                                       decorrelation_steps)
    H, xedges, yedges = np.histogram2d(samples[:, 0], samples[:, 1], bins=50)
    X, Y = np.meshgrid(xedges, yedges)
    fig = plt.figure()
    plt.pcolormesh(X, Y, H)
    fig.savefig('sample_test_2D.png')
