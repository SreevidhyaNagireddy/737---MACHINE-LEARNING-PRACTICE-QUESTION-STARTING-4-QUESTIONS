import numpy as np
from hmmlearn import hmm

# --- 1. Define the HMM Parameters ---

# We have 2 hidden states:
# State 0: Fair Die
# State 1: Loaded Die
n_states = 2

# We have 6 possible observations (emissions):
# 0 (roll 1), 1 (roll 2), ..., 5 (roll 6)
n_emissions = 6

# Initial State Probabilities:
# P(Fair) = 0.5, P(Loaded) = 0.5
start_probs = np.array([0.5, 0.5])

# Transition Probabilities: P(State_t | State_t-1)
#       To_Fair  To_Loaded
# From_Fair [ 0.95,    0.05    ]  (95% chance to stay Fair, 5% to switch)
# From_Loaded [ 0.10,    0.90    ]  (10% chance to switch to Fair, 90% to stay Loaded)
trans_matrix = np.array([
    [0.95, 0.05],
    [0.10, 0.90]
])

# Emission Probabilities: P(Observation | State)
#           Roll_1, Roll_2, Roll_3, Roll_4, Roll_5, Roll_6
# State_Fair  [ 1/6,   1/6,   1/6,   1/6,   1/6,   1/6   ]
# State_Loaded[ 0.1,   0.1,   0.1,   0.1,   0.1,   0.5   ] (50% chance of rolling a 6)
emission_probs = np.array([
    [1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]
])

# --- 2. Create the HMM Model ---
# We use MultinomialHMM because our observations are discrete (1-6)
model = hmm.CategoricalHMM(n_components=n_states, random_state=42)

# We are not training the model, but setting its parameters directly
model.startprob_ = start_probs
model.transmat_ = trans_matrix
model.emissionprob_ = emission_probs

# --- 3. Define an Observation Sequence ---
# Let's create a sequence of die rolls (0-5)
# Note the suspicious run of 5s (which means 6) at the end!
X = np.array([
    [0, 1, 5, 2, 4, 1, 3, 5, 0, 2,  # Mostly fair rolls
     5, 5, 3, 5, 1, 5, 5, 5, 4, 5]  # A suspicious number of 6s
]).T  # Transpose to make it a (n_samples, n_features) array

# --- 4. Predict the Hidden States ---
# Use the Viterbi algorithm to find the most likely hidden state sequence
log_prob, hidden_states = model.decode(X, algorithm="viterbi")

# --- 5. Print the Results ---
print(f"Observation sequence (0-5 maps to 1-6):")
# Add 1 to the observations for human-readable output (1-6)
print(" ".join(str(obs[0] + 1) for obs in X))
print("\nPredicted hidden states (0=Fair, 1=Loaded):")
print(" ".join(str(state) for state in hidden_states))

print(f"\nLog probability of this sequence: {log_prob:.2f}")

# A cleaner way to see the switch
print("\n--- Detailed Results ---")
print("Roll | State (0=Fair, 1=Loaded)")
print("--------------------------")
for i in range(len(X)):
    print(f" {X[i][0] + 1:^4} |  {hidden_states[i]:^20}")