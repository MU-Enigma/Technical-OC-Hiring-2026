# Project Proposal — The Locality Horizon: Predicting Spreading Influence

**Focus:** Network Dynamics & Graph Theory

---

## 1. Problem Statement

Predicting a node's spreading influence requires knowing how far ($r$ hops) one must look into the network. The apparent consensus in the field states that "you only need a few hops," noting that prediction quality becomes near-optimal when capturing orders up to K. 

However, this consensus conflates two entirely different quantities:
* **$I(r)$ (Information ceiling):** How much predictive information the $r$-ball physically contains, which is a property of the network and the dynamics.
* **$E(r)$ (Extraction):** How much a specific algorithm manages to pull out of that $r$-ball, which is a property of the chosen method

When a simple metric stops improving at $r \approx 3$, prior work measured $E(r)$ and misread it as the information limit $I(r)$. Measuring only one curve cannot determine whether the neighborhood ran out of information or if the metric simply hit its own algorithmic wall.

---

## 2. Objectives

1. **Separate the metrics:** Measure both the information ceiling $I(r)$ and the extracted signal $E(r)$ to isolate the gap $G(r) = I(r) - E(r).
2. **Test Conjecture H1:** Prove or falsify the hypothesis that the locality horizon $r^\star$ peaks at criticality ($\beta/\beta_c = 1$) because the correlation length .
3. **Prove bounds on ensembles:** Mathematically demonstrate that away from $\beta_c$, the horizon is $r^\star(\epsilon) = \mathcal{O}(\log 1/\epsilon)$ on locally tree-like ensembles.
4. **Build a deployable certificate:** Create a tool that uses a cheap local survey to predict whether local targeting is reliable or unreliable for a given network.

---

## 3. Expected Impact

**For the field.** The project reframes the methodology for evaluating influence predictors, moving away from metric leaderboards that conflate $I(r)$ and $E(r)$. It introduces a dynamic axis ($\beta/\beta_c$) to prove that the required observation radius is not fixed, but rather driven by distance from the critical point.

**For deployment.** Teams will possess a deployable certificate that uses local network properties to certify whether they can safely rely on local targeting or if distant structure matters too much.

---

## 4. Why It Is Worth Pursuing

* **The core hypothesis is explicitly falsifiable.** If the measured $r^\star$ remains flat across the transmissibility ratio $\beta/\beta_c$, Conjecture H1 is definitively wrong.
* **It relies on a known physical mechanism.** The assertion that $r^\star$ spikes at $\beta_c$ is physically motivated by the correlation length $\xi$, which dictates how far one node's fate stays coupled to another's. 
* **It isolates the true bottleneck.** By showing that linear scores go blind due to "echoes" (walks that backtrack and re-cross short loops), the project identifies *why* extraction fails even when the $r$-ball still holds predictive signal.
