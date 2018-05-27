# Proximal Policy Optimization (PPO)

## Overview

This is the code for [this](https://youtu.be/09OMoGqHexQ) video on Youtube by Siraj Raval.

PPO implementation for OpenAI Gym environment based on [Unity ML Agents](https://github.com/Unity-Technologies/ml-agents).

Notable changes include:

* Ability to continuously display progress with non-stochastic policy during training
* Works with OpenAI environments
* Option to record episodes
* State normalization for given number of frames
* Frame skip
* Faster reward discounting etc.

## Dependencies

* OpenAI Gym
* Tensorflow

## Run Program

```
python main.py
```

## Best Practices

[best_practices.md](docs\best_practices.md) contains guidelines for implementing PPO.


## Credits

* **Sven Niederberger** - [EmbersArc](https://github.com/EmbersArc) - *original author*
* **Siraj Raval** - [||Source||](https://github.com/llSourcell) - *contributor*
* **Benjamin Bueno** - [bbueno5000](https://github.com/bbueno5000) - *contributor*
