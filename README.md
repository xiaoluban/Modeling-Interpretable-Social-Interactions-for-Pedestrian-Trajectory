# Modeling-Interpretable-Social-Interactions-for-Pedestrian-Trajectory

## People interact with each other following different patterns. 

These interactive patterns can be represented as interpretable variables $[z^0, z^1, ..., z^{k-1}]$ with $k$ possibilities in interaction modeling.

## Model Framework

![m_methodology](https://github.com/Chogaliu/Interpretable-ped-Interaction/assets/80196339/34301528-5ea9-4032-87ba-53296fa0df4b)
Figure 2: Overview of our interaction modeling method. LSTM is used to capture the motion information of each agent. To incorporate social interaction in generating the next state at time $t+1$, we use the relative motion between an agent and its neighbors, represented as $(\mathcal{I}_i^t- \mathcal{I}_j^t)$, to derive a latent representation of the social interaction, $s_i^t$. Our method consists of two stages: mode extraction and mode aggregation. In the mode extraction stage, we take the relative motion information as input and encode it with past information represented by hidden states to generate mode embeddings. For each interaction, we sample a possible mode representation $z_j^t$ from the generated mode embeddings. Then the representation of modalized interaction can be achieved by $(\mathcal{I}_i^t- \mathcal{I}_j^t)$ and $z_j^t$. In the mode aggregation stage, we sort up all interactions into different groups based on their modes. The weight $a_j^t$ for interaction $j$ between the same group is calculated using softmax. And $s_i^t$ is the sum of the weighted representations from each group.


## Documentation
- **AppleStore_ECE.m**:
- **initial_pedes_co.mat**: 


## Results  

Using an actual evacuation video under extreme panic as evidence, the characteristics of pedestrian behavior under panic were analyzed, and the ECE model was demonstrated to be accurate for predicting evacuation efficiency. Furthermore, it was found to reproduce the individual movements of pedestrians (detouring behavior and the “parallel to single” phenomenon) better than the social force model. 

| Model name | RouteDeviation |
| :---: | :---: | 
| ECE | 1.2660 |
| SFM(v=5) | 2.1243 |
| SFM(v=4) | 1.7170 |
| SFM(v=3) | 1.4529 |
| SFM(v=2) | 1.4888 |
