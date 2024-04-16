```diff
- text in red
+ text in green
! text in orange
# text in gray
@@ text in purple (and bold)@@
```

# Interpretable-ped-Interaction

## People interact with each other following different patterns. 

These interactive patterns can be represented as interpretable variables $[z^0, z^1, ..., z^{k-1}]$ with $k$ possibilities in interaction modeling.

## Paper Details

- Title : Modeling Interpretable Social Interactions for Pedestrian Trajectory
- Abstract : The abilities to understand pedestrian social interaction behaviors and to predict their future trajectories are critical for road safety, traffic management and more broadly autonomous vehicles and robots.
Social interactions are intuitively heterogeneous and dynamic over time and circumstances, making them hard to explain.@@
In this paper, we creatively investigate modeling interpretable social interactions for pedestrian trajectory, which is not considered by the existing trajectory prediction research. Moreover, we propose a two-stage methodology for interaction modeling - "mode extraction" and "mode aggregation", and develop a long short-term memory (LSTM)-based model for long-term trajectory prediction, which naturally takes into account multi-types of social interactions. ${\texttt{\color{red}Different from previous models that don’t explain how pedestrians interact socially, we extract latent modes that represent social interaction types which scales to an arbitrary number of neighbors.}}$ Extensive experiments over two public datasets have been conducted. The quantitative and qualitative results demonstrate that our method is able to capture the multi-modality of human motion and achieve better performance under specific conditions. Its performance is also verified by the interpretation of predicted modes, of which the results are in accordance with common sense. Besides, we have performed sensitivity analysis on the crucial hyperparameters in our model.

+ State : Accepted by Transportation Research Part C: Emerging Technologie.
+ Link : xxxx http:

## Model Framework

![m_methodology](https://github.com/Chogaliu/Interpretable-ped-Interaction/assets/80196339/34301528-5ea9-4032-87ba-53296fa0df4b)
Figure 2: Overview of our interaction modeling method. LSTM is used to capture the motion information of each agent. To incorporate social interaction in generating the next state at time $t+1$, we use the relative motion between an agent and its neighbors, represented as $(\mathcal{I}_i^t- \mathcal{I}_j^t)$, to derive a latent representation of the social interaction, $s_i^t$. Our method consists of two stages: mode extraction and mode aggregation. In the mode extraction stage, we take the relative motion information as input and encode it with past information represented by hidden states to generate mode embeddings. For each interaction, we sample a possible mode representation $z_j^t$ from the generated mode embeddings. Then the representation of modalized interaction can be achieved by $(\mathcal{I}_i^t- \mathcal{I}_j^t)$ and $z_j^t$. In the mode aggregation stage, we sort up all interactions into different groups based on their modes. The weight $a_j^t$ for interaction $j$ between the same group is calculated using softmax. And $s_i^t$ is the sum of the weighted representations from each group.


## Documentation
- **AppleStore_ECE.m**: The code is to simulate the dynamic of pedestrain using ECE model (case study)
- **AppleStore_SFM.m**: The code is to simulate the dynamic of pedestrain using social force model (case study)
- **initial_pedes_co.mat**: The initial position of evacuees based on the field video [ped_id, x, y]


## Results  

Using an actual evacuation video under extreme panic as evidence, the characteristics of pedestrian behavior under panic were analyzed, and the ECE model was demonstrated to be accurate for predicting evacuation efficiency. Furthermore, it was found to reproduce the individual movements of pedestrians (detouring behavior and the “parallel to single” phenomenon) better than the social force model. 

| Model name | RouteDeviation |
| :---: | :---: | 
| ECE | 1.2660 |
| SFM(v=5) | 2.1243 |
| SFM(v=4) | 1.7170 |
| SFM(v=3) | 1.4529 |
| SFM(v=2) | 1.4888 |
