import numpy as np
from sklearn.cluster import DBSCAN
from pyransac3d import Plane
from pyransac3d import Cylinder
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

# Klasteryzacja DBSCAN
def cluster_with_dbscan(points, eps=0.3, min_samples=30):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(points)
    clusters = []
    for label in set(labels):
        if label == -1:
            continue  # pomin szum
        clusters.append(points[labels == label])
    print(f"Znaleziono {len(clusters)} klastrow (bez szumu).")
    return clusters, labels

# Dopasowanie plaszczyzny z pyransac3d
def analyze_cluster_with_pyransac3d(cluster, cluster_id):
    print(f"\nAnaliza klastra {cluster_id + 1}, punktow: {len(cluster)}")

    plane = Plane()
    is_plane = False
    try:
        model, inliers = plane.fit(cluster, thresh=0.01)
        if model is None or len(inliers) == 0:
            print("Nie udało się dopasować płaszczyzny.")
        else:
            a, b, c, d = model
            inlier_ratio = len(inliers) / len(cluster)
            print(f"Plane normal: [{a:.3f}, {b:.3f}, {c:.3f}], inliers: {len(inliers)} / {len(cluster)} ({inlier_ratio:.2f})")

            if inlier_ratio > 0.9:
                print("To prawdopodobnie plaszczyzna.")
                is_plane = True  # Uznaj klaster za płaszczyznę
                if abs(c) > 0.9:
                    print("Plaszczyzna pozioma.")
                elif abs(a) > 0.9 or abs(b) > 0.9:
                    print("Plaszczyzna pionowa.")

    except Exception as e:
        print(f"Blad przy dopasowaniu plaszczyzny: {e}")

    # Dopasowanie cylindra tylko jeśli nie jest płaszczyzną
    if not is_plane:
        try:
            cyl = Cylinder()
            model_c, direction, radius, inliers_c = cyl.fit(cluster, thresh=0.05)
            if inliers_c is None or len(inliers_c) == 0:
                print("Nie udalo sie dopasowac cylindra.")
            else:
                inlier_ratio_cyl = len(inliers_c) / len(cluster)
                if inlier_ratio_cyl > 0.8:
                    print("To prawdopodobnie cylinder.")
                    print(f"Model cylindra: Parametry: {model_c}, Kierunek: {direction}, Promien: {radius}")
                else:
                    print("Nie wyglada na cylinder.")
        except Exception as e:
            print(f"Blad przy dopasowaniu cylindra: {e}")



def plot_dbscan_clusters(points, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.colormaps['tab10']
    for i, label in enumerate(set(labels)):
        if label == -1:
            continue
        pts = points[labels == label]
        color = colors(i / 9)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, color=color, label=f'Cluster {label}')
    ax.legend()
    ax.set_title("DBSCAN - podzial na klastry")
    plt.show()

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


# Wczytaj 3 pliki
file1 = load_xyz_file("zad1/flat_surface.xyz")
file2 = load_xyz_file("zad1/vertical_surface.xyz")
file3 = load_xyz_file("zad1/cylindrical_surface.xyz")

# Polaczenie w jedna chmure
points = np.vstack((file1, file2, file3))
print("Wszystkie punkty:", points.shape)
clusters, labels = cluster_with_dbscan(points, eps=0.85, min_samples=50)
plot_dbscan_clusters(points, labels)

# Analiza kazdego klastra
for i, cluster in enumerate(clusters):
    analyze_cluster_with_pyransac3d(cluster, i)
