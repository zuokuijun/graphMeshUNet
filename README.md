# GraphMeshUNet: A Physics-Guided Graph Autoencoder for High-Fidelity Three Dimensional Golf Ball Shape Generation


<p align="center">
  <img src="./images/Golf ball.png" alt="graphMeshUNet" width="400">
</p>

<p align="center">
  <img src="./images/graphMeshUNet.png" alt="graphMeshUNet" width="800">
</p>

### Overall architecture of GraphMeshUNet for 3D golf ball shape generation. (a) Input mesh represented as a graph with node features, edge features, and an adjacency matrix. (b) Encoder-decoder architecture with multiscale skip connections.(c) Encoding and decoding blocks with GINE-based feature extraction, graph pooling, index-based upsampling, and skip-feature fusion. (d) Drag coefficient encoder and GINE block module.

<p align="center">
  <img src="./images/dataset.png" alt="graphMeshUNet" width="800">
</p>

### Golf ball datasets used for training GraphMeshUNet. (a) GolfAero-Synth generated using Blender, each synthetic golf ball is identified by the prefix \enquote{Golf} followed by three numerical fields representing $n_d$, $r_g$, and $k_d$, respectively. (b) GolfAero-Real acquired by 3D scanning commercial golf balls, each commercial golf ball is identified by the prefix \enquote{Ball} followed by two numerical fields: the ball identifier and its drag coefficient, respectively.


