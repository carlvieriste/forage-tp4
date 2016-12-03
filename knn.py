import numpy as np
from scipy.spatial.distance import pdist, squareform

print("Load ratings")
# Load Z normalized ratings data
N = 943
M = 1682
r_ui = np.zeros((N, M))
with open("../ratedMatrix.txt", "r") as file:
    for line in file.readlines():
        tokens = line.strip().split(' ')
        u = int(tokens[0])
        i = int(tokens[1])
        r = float(tokens[2].replace(',', '.'))
        r_ui[u, i] = r

print("Load has_rated")
# Load similarity data
has_rated = np.zeros(r_ui.shape)
with open("../userMovieRated.txt", "r") as file:
    for line in file.readlines():
        tokens = line.strip().split(' ')
        u = int(tokens[0])
        i = int(tokens[1])
        did_rate = int(tokens[2])
        has_rated[u, i] = did_rate

print("Load mean, stddev")
# Load test data
user_mean = np.zeros((N,))
user_stddev = np.zeros((N,))
with open("../moyStdev.txt", "r") as file:
    for u, line in enumerate(file.readlines()):
        tokens = line.strip().split(' ')
        mean = float(tokens[0].replace(',', '.'))
        stddev = float(tokens[1].replace(',', '.'))
        user_mean[u] = mean
        user_stddev[u] = stddev

print("Load test data")
# Load test data
test_data = np.genfromtxt("../u1.test", delimiter="\t", usecols=(0, 1, 2))
# with open("../u1.test", "r") as file:
#     for line in file.readlines():
#         tokens = line.strip().split('\t')
#         u = int(tokens[0])
#         i = int(tokens[1])
#         rating = int(tokens[2])
#         test_data[u, i] = rating
#
# np.savetxt("fml.txt", test_data[0:100])

# raw_data = np.genfromtxt("../u1.base", delimiter="\t", usecols=(0, 1, 2))  # 3 columns : user_id, movie_id, rating
# rating = np.genfromtxt("../ratedMatrix.txt", delimiter=" ")
# has_rated = np.genfromtxt("../userMovieRated.txt", delimiter=" ")

# np.savetxt("out.csv", r_ui[:1000])
print("Compute similarity")
# Compute similarity
distance_condensed = pdist(r_ui, metric='wminkowski', p=2, w=has_rated)
distance = squareform(distance_condensed)
print(distance.shape)
w_uv = 1.0 / (0.0001 + distance)

print("Adjust similarity according to commonly rated movies")
# Adjust similarity according to commonly rated movies
beta = 4
I_uv = np.dot(has_rated, has_rated.transpose())
print(I_uv.shape)
print(w_uv.shape)
w_prime = I_uv / (I_uv + beta) * w_uv

print("Sort by similarity")
sorted_by_simil = np.argsort(w_prime, axis=1)[:, ::-1]

K = 20
err = 0
print("Total number of pairs:", len(test_data))
compare = np.zeros((len(test_data), 2))
num_not_enough = 0
num_not_similar = 0
for i, user_movie_pair in enumerate(test_data):
    # print(i)

    user_id = int(user_movie_pair[0])
    movie_id = int(user_movie_pair[1])
    real_rating = user_movie_pair[2]

    # Find K closest users
    k_closest_users = []
    for close_user in sorted_by_simil[user_id]:
        if has_rated[close_user, movie_id]:
            k_closest_users.append(close_user)
        if len(k_closest_users) == K:
            break

    if len(k_closest_users) > 0:
        # Estimate rating
        numerateur = 0
        denominateur = 0
        for u in k_closest_users:
            numerateur += w_prime[user_id, u] * r_ui[u, movie_id]
            denominateur += w_prime[user_id, u]

        if denominateur > 0.001:
            estimated_z = numerateur / denominateur
            estimated = estimated_z * user_stddev[user_id] + user_mean[user_id]
        else:
            estimated = real_rating  # trololo
            num_not_similar += 1
    else:
        estimated = real_rating  # lol
        num_not_enough += 1

    err += np.abs(real_rating - estimated)
    compare[i] = (real_rating, estimated)

print("Erreur moyenne en etoiles:", err / len(test_data))
print("Cas pas assez de voisins:", num_not_enough)
print("Cas pas assez similaires:", num_not_similar)
np.savetxt("predict.csv", compare)