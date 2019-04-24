"""
k-means clustering algorithm
Step 1 - Pick K random points as cluster centers called centroids.
Step 2 - Assign each x to nearest cluster by calculating its distance to each centroid.
Step 3 - Find new cluster center by taking the average of the assigned points.
Step 4 - Repeat Step 2 and 3 until none of the cluster assignments change.
"""
import random
import matplotlib.pyplot as plt

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']


def k_means(data, k, iter):
    # Pick k random points
    rand_points = _k_rand_points(data, k)

    plt.ion()
    for i in range(iter):
        result = [[] for _ in range(k)]
        for point in data:
            position = _min_distance(point, rand_points)
            result[position].append(point)

        # Calculate new cluster center
        rand_points = _centroids(result, k)

        # show plots
        for j in range(k):
            x = [p[0] for p in result[j]]
            y = [p[1] for p in result[j]]
            plt.scatter(x, y, c=COLORS[j%len(COLORS)])
        plt.title(str(i)+' iterations')
        plt.draw()
        plt.pause(0.1)
    plt.ioff()

    return result


def _centroids(result, k):
    new_centers = []
    for j in range(k):
        new_centers.append(
            (sum([i[0] for i in result[j]]) / len(result[j]), sum([i[1] for i in result[j]]) / len(result[j])))
    return new_centers


def _min_distance(point, rand_points):
    dis = [_distance(point, p) for p in rand_points]
    return dis.index(min(dis))


def _k_rand_points(data, k):
    if k > len(data):
        raise Exception('k should be smaller than the length of data.')
    rand_points = []
    while len(rand_points) < k:
        point = random.choice(data)
        if point not in rand_points:
            rand_points.append(point)
    return rand_points


def _distance(x, y):
    return (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2


def generate_data(num):
    data1 = [(random.random() * 10, random.random() * 10) for _ in range(num)]
    data2 = [(random.random() * 10 + 10, random.random() * 10 + 10) for _ in range(num)]
    data3 = [(random.random() * 10 + 20, random.random() * 10) for _ in range(num)]
    return data1 + data2 + data3


if __name__ == '__main__':
    data = generate_data(500)
    k_means(data, 3, 50)
