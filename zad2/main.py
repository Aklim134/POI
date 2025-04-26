import numpy as np
import random
import csv
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def plane_from_points(p1, p2, p3):
    # Dwa wektory leżace na plaszczyźnie
    v1 = p2 - p1
    v2 = p3 - p1

    # Wektor normalny
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)  # normalizacja

    # Rownanie plaszczyzny: ax + by + cz + d = 0
    # gdzie [a, b, c] to normalny, d = -normal . p1
    d = -np.dot(normal, p1)
    return normal, d

def point_plane_distance(point, normal, d):
    return abs(np.dot(normal, point) + d) / np.linalg.norm(normal)

def ransac_plane_fitting(points, threshold=0.01, iterations=300):
    best_inliers = []
    best_plane = (None, None)

    for _ in range(iterations):
        # Wybierz losowo 3 rozne punkty
        sample = points[np.random.choice(points.shape[0], 3, replace=False)]
        p1, p2, p3 = sample

        try:
            normal, d = plane_from_points(p1, p2, p3)
        except:
            continue

        inliers = []
        for point in points:
            if point_plane_distance(point, normal, d) < threshold:
                inliers.append(point)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = (normal, d)

    return best_plane, np.array(best_inliers)

def load_xyz_file(file_path):
    points = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            if len(row) != 3:
                continue  # pomiń bledne linie
            try:
                x, y, z = map(float, row)
                points.append([x, y, z])
            except ValueError:
                continue
    return np.array(points)

def analyze_cluster(cluster_points, cluster_id):
    print(f"\nAnaliza klastra {cluster_id + 1}:")
    (normal, d), inliers = ransac_plane_fitting(cluster_points, threshold=0.01)

    print(f"Wektor normalny: {normal}")
    print(f"Parametr d: {d}")
    print(f"Liczba punktow dopasowanych (inliers): {len(inliers)}")

    # Oblicz srednia odleglosc wszystkich punktów do plaszczyzny
    distances = [point_plane_distance(p, normal, d) for p in cluster_points]
    mean_distance = np.mean(distances)

    print(f"srednia odleglosc punktow od plaszczyzny: {mean_distance:.6f}")

    # Okreslenie typu powierzchni
    if mean_distance < 0.01:
        if np.abs(normal[2]) > 0.9:
            print("To jest plaszczyzna pozioma.")
        elif np.abs(normal[0]) > 0.9 or np.abs(normal[1]) > 0.9:
            print("To jest plaszczyzna pionowa.")
        else:
            print("To jest plaszczyzna, ale pod katem.")
    else:
        print("To prawdopodobnie nie jest plaszczyzna.")


# Wczytaj 3 pliki
file1 = load_xyz_file("zad1/flat_surface.xyz")
file2 = load_xyz_file("zad1/vertical_surface.xyz")
file3 = load_xyz_file("zad1/cylindrical_surface.xyz")

# Polaczenie w jedna chmure
points = np.vstack((file1, file2, file3))
print("Wszystkie punkty:", points.shape)

k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(points)

# Podzial punktów na klastry
clusters = [points[labels == i] for i in range(k)]

# Analiza każdego klastra
for i, cluster in enumerate(clusters):
    analyze_cluster(cluster, i)

# Szybka wizualizacja (rzut na XY)
colors = ['red', 'green', 'blue']
for i in range(k):
    cluster = clusters[i]
    plt.scatter(cluster[:, 0], cluster[:, 1], s=1, color=colors[i], label=f'Cluster {i+1}')
plt.title("KMeans clustering (rzut XY)")
plt.legend()
plt.show()