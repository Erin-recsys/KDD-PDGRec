import torch


def calculate_similarity(preferences, social_graph):
    users = list(preferences.keys())

    similarity_dic = {user: {} for user in users}

    social_edges = social_graph.edges()
    social_edges_dict = {(social_edges[0][i].item(), social_edges[1][i].item()) for i in range(len(social_edges[0]))}
    print(social_edges_dict)

    for i in range(len(users)):
        user_i = users[i]
        print(user_i)
        for j in range(i + 1, len(users)):
            user_j = users[j]

            if (user_i, user_j) in social_edges_dict or (user_j, user_i) in social_edges_dict:
                prefs_i = preferences[user_i]
                prefs_j = preferences[user_j]
                print('==============')
                abs_diff = 0
                for genre in prefs_i.keys():
                    if genre in prefs_j:
                        abs_diff += abs(prefs_i[genre] - prefs_j[genre])

                similarity = 1 - (abs_diff / len(prefs_i))
                similarity_dic[user_i][user_j] = similarity
                similarity_dic[user_j][user_i] = similarity

    return similarity_dic


def assign_weights_to_relationships(preferences, social_graph):
    similarity_dic = calculate_similarity(preferences, social_graph)
    weights = []

    social_edges = social_graph.edges()
    #print(social_edges)
    for i in range(len(social_edges[0])):
        user_i = social_edges[0][i].item()
        user_j = social_edges[1][i].item()
        #print(user_i, user_j)

        if user_i in similarity_dic and user_j in similarity_dic[user_i]:
            weight = similarity_dic[user_i][user_j]
        else:
            weight = 0

        weights.append(weight)
        print(weights)
    social_graph.edata['weight'] = torch.tensor(weights, dtype=torch.float32)

    return social_graph
