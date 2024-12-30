## What and Why State-Space Model is Important:

SSMs help estimate hidden true states (e.g., car position) from noisy observations, making them crucial in control systems, robotics, and financial modeling. 

Use cases include:

- `Self-driving cars`: Tracking vehicles and pedestrians.
- `Weather forecasting`: Estimating future climate states.
- `Economics`: Modeling market trends.

### Real-Time Example: Tracking a Moving Car

Key Terms:

- **State Transition**: The car's true position evolves over time, e.g., position(t+1) = position(t) + speed + noise.

- **Observation Model**: GPS measures the car's position but includes noise.

- **True State**: The actual position of the car (unknown but estimated).

- **Next State**: The predicted position of the car based on the model.

- **Observation**: The noisy GPS reading of the car’s position.

### Pseudo Algorithm: Car Tracking Using State-Space Model

1. **Initialize Parameters**:

- Set initial position (true_state[0]).
- Define constants like velocity, transition noise, and observation noise.

2. **For each time step (from 1 to total steps)**:

- State Transition:
    - Compute the next true position:
    `next_state = current_state + velocity + transition_noise`
    - Append next_state to true_state.
- Observation Model:
    - Simulate noisy measurement:
    `observation = next_state + observation_noise`
    - Store observation.

3. **Output Results**:

- Return the list of true_state and observations.

4. **Optional Visualization**:

- Plot true_state (actual positions).
- Plot observations (noisy GPS readings).

**Key Notes**:

- The algorithm predicts the system's evolution (state transition) and generates noisy observations to mimic real-world measurements.
Adjusting noise parameters or adding Bayesian inference can refine the accuracy of predictions.


### State Transitions, Observations in short:

**`Extrapolation`**: Using the true position and state transition formula, we predict the next state and add noise to simulate real-world uncertainty.

**`Observations`**: Represent noisy measurements (e.g., GPS) around the predicted state(which comes from state transition).

**`Inference`**: By combining predictions and noisy observations, we estimate the future behavior of the system. This is state transition prediction + uncertainity observations around it.

>Note: If the true position or state transition formula is inaccurate or poorly designed, predictions and observations will deviate significantly from reality, leading to erroneous inferences. This highlights the importance of accurately modeling both the state evolution and observation processes for reliable predictions.

>Note: Bayesian posterior inference can estimate state transition parameters and observation noise by updating beliefs about these parameters based on observed data. This helps refine the model for more accurate predictions and state estimations.