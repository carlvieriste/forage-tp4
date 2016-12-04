import numpy as np
from scipy.spatial.distance import pdist, squareform


def load_data():
    print("Load ratings")
    ratings = np.zeros((N, M))
    with open("../ratedMatrix.txt", "r") as file:
        for line in file.readlines():
            tokens = line.strip().split(' ')
            u = int(tokens[0])
            i = int(tokens[1])
            r = float(tokens[2].replace(',', '.'))
            ratings[u, i] = r

    print("Load has_rated")
    # Load similarity data
    has_rated = np.zeros(ratings.shape)
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

    return ratings, has_rated, user_mean, user_stddev


def preprocess(set_name):
    print("Preprocess")

    # Load base data
    data_raw = np.genfromtxt("../" + set_name + ".base", delimiter="\t", usecols=(0, 1, 2))

    # Make r_ui matrix (1-5 ratings)
    data = np.zeros((N, M))
    for rate_pair in data_raw:
        u = int(rate_pair[0]) - 1
        i = int(rate_pair[1]) - 1
        r = rate_pair[2]
        data[u, i] = r

    has_rated = (data >= 1)
    has_not_rated = np.logical_not(has_rated)
    nandata = data
    nandata[has_not_rated] = np.nan

    user_mean = np.nanmean(nandata, axis=1)
    user_stddev = np.nanstd(nandata, axis=1)

    # Get Z-score
    r_ui = (data - user_mean[:, None]) / user_stddev[:, None]
    r_ui[has_not_rated] = 0

    return r_ui, has_rated, user_mean, user_stddev


# Load Z normalized ratings data
N = 943
M = 1682
# set_name = "u1"
for set_name in ['u1', 'u2', 'u3', 'u4', 'u5']:
    r_ui, has_rated, user_mean, user_stddev = preprocess(set_name)  # preprocess load_data

    print("Load test data")
    # Load test data
    test_data = np.genfromtxt("../" + set_name + ".test", delimiter="\t", usecols=(0, 1, 2))

    print("Compute similarity")
    # Compute similarity
    distance_condensed = pdist(r_ui, metric='wminkowski', p=2, w=has_rated)
    distance = squareform(distance_condensed)
    w_uv = 1.0 / (0.0001 + distance)

    print("Adjust similarity according to commonly rated movies")
    # Adjust similarity according to commonly rated movies
    beta = 4
    I_uv = np.dot(has_rated, has_rated.transpose())
    w_prime = I_uv / (I_uv + beta) * w_uv

    print("Sort by similarity")
    sorted_by_simil = np.argsort(w_prime, axis=1)[:, ::-1]

    K = 20
    err_MAE_sum = 0
    print("Total number of pairs:", len(test_data))
    compare = np.zeros((len(test_data), 2))
    num_not_enough = 0
    num_not_similar = 0
    for i, user_movie_pair in enumerate(test_data):
        # print(i)

        user_id = int(user_movie_pair[0]) - 1
        movie_id = int(user_movie_pair[1]) - 1
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

        err_MAE_sum += np.abs(real_rating - estimated)
        compare[i] = (real_rating, estimated)

    total_valid = len(test_data) - num_not_enough - num_not_similar

    MAE_quality = err_MAE_sum / total_valid

    print("MAE Quality Measure:", MAE_quality)
    print("Cas pas assez de voisins:", num_not_enough)
    print("Cas pas assez similaires:", num_not_similar)
    np.savetxt("predict.csv", compare)

    with open("cumulative_results.txt", "a") as file:
        file.write("\nDataset: {}\n".format(set_name))
        file.write("MAE Quality: {}\n".format(MAE_quality))
        file.write("Cases with zero valid neighbors:   {}\n".format(num_not_enough))
        file.write("Cases with zero similar neighbors: {}\n".format(num_not_similar))

        # Erreur moyenne en etoiles: 1.04099029627
        # Cas pas assez de voisins: 54
        # Cas pas assez similaires: 6
