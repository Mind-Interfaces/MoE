import torch
import torch.nn as nn
import torch.nn.functional as F
class SimpleExpert(nn.Module):
    def __init__(self):
        super(SimpleExpert, self).__init__()
        self.fc1 = nn.Linear(10, 10)

    def forward(self, x):
        return F.relu(self.fc1(x))
class GatingMechanism(nn.Module):
    def __init__(self, num_experts):
        super(GatingMechanism, self).__init__()
        self.fc = nn.Linear(10, num_experts)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=1)
class MixtureOfExperts(nn.Module):
    def __init__(self, num_experts):
        super(MixtureOfExperts, self).__init__()
        self.experts = nn.ModuleList([SimpleExpert() for _ in range(num_experts)])
        self.gating = GatingMechanism(num_experts)

    def forward(self, x):
        # Get gating weights
        gating_weights = self.gating(x)

        # Forward pass through each expert
        expert_outputs = [expert(x) for expert in self.experts]

        # Combine the experts' outputs based on the gating weights
        combined_output = 0
        for i, expert_output in enumerate(expert_outputs):
            combined_output += gating_weights[:, i:i+1] * expert_output

        return combined_output
# Number of experts
num_experts = 16

# Instantiate MoE
moe = MixtureOfExperts(num_experts)

# Sample input
x = torch.rand(5, 10)

# Forward pass
output = moe(x)
print(output)
