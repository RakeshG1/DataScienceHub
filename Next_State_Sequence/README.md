# Next State & Sequence prediction models

## Markov Chain Model:

**Assumption**: The next state depends only on the current state (memoryless property).

**Use Cases**: Weather prediction, text generation, customer behavior modeling, where future states primarily depend on the immediate past.

A Markov Chain is a mathematical model that describes a system transitioning between states, where the next state depends only on the current state (not the past history). This is known as the Markov property.

### How it Works Internally?

**States**: The system has a set of possible states.

**Transition Probabilities**: The probability of moving from one state to another is defined in a transition matrix.

**No Memory**: The future state depends only on the current state, making it "memoryless."

Use in ML and Bayesian Applications

**Machine Learning**: Used in sequence modeling (e.g., Hidden Markov Models for speech recognition, NLP).

**Bayesian Inference**: In Bayesian applications, Markov Chain Monte Carlo (MCMC) uses Markov Chains to sample from the posterior distribution when direct computation is challenging.

>Note: Primarily a next-state prediction model but can be extended for simple sequence predictions.

## Hidden Markov Model (HMM): 

Includes hidden/latent states, making it suitable for problems like speech recognition or part-of-speech tagging.

In an HMM, hidden/latent states are unobservable internal states (e.g., emotions in speech) that influence observable outcomes (e.g., spoken words). The model predicts these hidden states based on observed data and transitions.

>Note: Used for sequence prediction, especially when sequences depend on hidden states (e.g., speech recognition).

## LSTM (Long Short-Term Memory): 

Considers longer dependencies and complex patterns, ideal for time-series forecasting or sequence problems like stock price prediction.

>Note: Designed for sequence prediction, capturing long-term dependencies in data (e.g., text or time-series).




