import numpy as np

def compute_intensities(model,t, sequence):
    intensity = np.zeros(model.num_types)
    for c in range(model.num_types):
        intensity[c] = model.compute_intensity(t, c, sequence)

    return intensity

def ogata_thinning_algorithm(model,t_j,sequence,l_t=1e10000):
    t = t_j
    while 1:
        # Step 2(a): Compute m(t) and l(t)
        m_t = compute_intensities(model,t,sequence).sum()  # Function to compute m(t)
        l_t = l_t  # Function to compute l(t)

        # Step 2(b): Generate random variables
        s = np.random.exponential(m_t)  # Generate s ~ Exp(m(t))
        U = np.random.uniform()         # Generate U ~ Unif([0, 1])

        if s > l_t:
            t = t + l_t
            pred_time = t + l_t
            break
        elif U > (compute_intensities(model,t+s,sequence).sum() / m_t):
            t = t + s
        else:
            pred_time = t + s
            break
    return pred_time


def predict_time(model,t_j,sequence,num_iterations=2):
    pred_times = np.zeros(num_iterations)
    for i in range(num_iterations):
        pred_times[i] = ogata_thinning_algorithm(model,t_j,sequence)
    return np.mean(pred_times)

def predict(model,sequences,num_iterations):
    pred_events = []
    for a, sequence in enumerate(sequences):
        pred_event = []
        for m, (t_m, c_m) in enumerate(sequence):
            pred_time = predict_time(model,t_m,sequence,num_iterations)

            intensity = compute_intensities(model,pred_time,sequence)

            pred_type = np.argmax(intensity)

            pred_event.append((pred_time,pred_type))

        pred_events.append(pred_event)

    return pred_events

