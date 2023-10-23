import torch
import torch.nn as nn
import torch.nn.functional as fun


class SimpleExpert(nn.Module):
    def __init__(self):
        super(SimpleExpert, self).__init__()
        self.fc1 = nn.Linear(10, 10)

    def forward(self, user_input):
        return fun.relu(self.fc1(user_input))


class GatingMechanism(nn.Module):
    def __init__(self, num_experts):
        super(GatingMechanism, self).__init__()
        self.fc = nn.Linear(10, num_experts)

    def forward(self, user_input):
        return fun.softmax(self.fc(user_input), dim=1)


class MixtureOfExperts(nn.Module):
    def __init__(self, num_experts):
        super(MixtureOfExperts, self).__init__()
        self.experts = nn.ModuleList([SimpleExpert() for _ in range(num_experts)])
        self.gating = GatingMechanism(num_experts)

    def forward(self, user_input):
        # Get gating weights
        gating_weights = self.gating(user_input)

        # Forward pass through each expert
        expert_outputs = [expert(user_input) for expert in self.experts]

        # Combine the experts' outputs based on the gating weights
        combined_output = 0
        for i, expert_output in enumerate(expert_outputs):
            combined_output += gating_weights[:, i:i+1] * expert_output

        return combined_output


# Number of experts
num_experts = 16

# Instantiate MoE
moe = MixtureOfExperts(num_experts)

# Sample input for example. Batch size = 5, input size = 10
user_input = torch.rand(5, 10)

# Forward pass
output = moe(input)
print(output)
