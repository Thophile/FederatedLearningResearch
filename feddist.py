from joblib import dump, load
import numpy as np

def roger_federer(LOCAL_MODELS, HIDDEN_LAYER_SIZE):
    print("---------- Computing avg ----------")
    # Compute averages
    coefs_avg = np.array(LOCAL_MODELS[0].coefs_, dtype=object)
    intercepts_avg = np.array(LOCAL_MODELS[0].intercepts_, dtype=object)
    for model in LOCAL_MODELS[1:]:
        coefs_avg += np.array(model.coefs_, dtype=object)
        intercepts_avg += np.array(model.intercepts_, dtype=object)

    coefs_avg = np.true_divide(coefs_avg, len(LOCAL_MODELS))
    intercepts_avg = np.true_divide(intercepts_avg, len(LOCAL_MODELS))

    print("---------- Computing sigmas ----------")
    # Compute sigmas
    coefs_sigma = np.square(np.array(LOCAL_MODELS[0].coefs_, dtype=object) - np.array(coefs_avg, dtype=object))
    intercepts_sigma = np.square(np.array(LOCAL_MODELS[0].intercepts_, dtype=object) - np.array(intercepts_avg, dtype=object))
    for model in LOCAL_MODELS[1:]:
        coefs_sigma += np.square(np.array(model.coefs_, dtype=object) - np.array(coefs_avg, dtype=object))
        intercepts_sigma += np.square(np.array(model.intercepts_, dtype=object) - np.array(intercepts_avg, dtype=object))

    coefs_sigma = (np.true_divide(coefs_sigma, len(LOCAL_MODELS))) ** 0.5
    intercepts_sigma = (np.true_divide(intercepts_sigma, len(LOCAL_MODELS))) ** 0.5

    # Declaration of new values
    new_coefs = coefs_avg
    new_intercepts = intercepts_avg
    NEW_LAYER_SIZE = list(HIDDEN_LAYER_SIZE)

    # For each layer
    for layer in range(len(HIDDEN_LAYER_SIZE)):
        print("---------- Federating Layer " + str(layer + 1) + "/" + str(len(HIDDEN_LAYER_SIZE)) + "----------")

        layer_coefs_avg = coefs_avg[layer + 1]
        layer_intercepts_avg = intercepts_avg[layer]
        layer_coefs_sigma = coefs_sigma[layer + 1]
        layer_intercepts_sigma = intercepts_sigma[layer]

        # for model in LOCAL_MODELS:
        #     for percetron_index in range (len(model.coefs_[layer + 1])):
        #         coefs_node = model.coefs_[layer + 1][percetron_index]
        #         intercepts_node = model.intercepts_[layer][percetron_index]
        #         for node_index in range (len(model.coefs_[layer + 1][percetron_index])):
        #             if (coefs_node[node_index] > layer_coefs_avg[percetron_index][node_index] + 10*layer_coefs_sigma[percetron_index][node_index] 
        #             or coefs_node[node_index] < layer_coefs_avg[percetron_index][node_index] - 10*layer_coefs_sigma[percetron_index][node_index]):
        #                 np.append(new_coefs[layer + 1], coefs_node)
        #                 np.append(new_intercepts[layer], intercepts_node)
        #                 NEW_LAYER_SIZE[layer] += 1
        #                 break

        for model in LOCAL_MODELS:
            for percetron_index in range (len(model.coefs_[layer + 1])):
                coefs_node = model.coefs_[layer + 1][percetron_index]
                intercepts_node = model.intercepts_[layer][percetron_index]
                if (intercepts_node > layer_intercepts_avg[percetron_index] + layer_intercepts_sigma[percetron_index] 
                or intercepts_node < layer_intercepts_avg[percetron_index] - layer_intercepts_sigma[percetron_index]):
                    np.append(new_coefs[layer + 1], coefs_node)
                    np.append(new_intercepts[layer], intercepts_node)
                    NEW_LAYER_SIZE[layer] += 1
                        
    return tuple(NEW_LAYER_SIZE), new_coefs, new_intercepts

