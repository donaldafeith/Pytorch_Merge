# PyTorch Merge
# Copyright (C) 2023 Donalda Feith
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
from transformers import AutoConfig

def main():
    # Prompt user for input paths
    config_path = input("Enter the path to the config.json file: ")
    model1_path = input("Enter the path to the first model .bin file (model1.bin): ")
    model2_path = input("Enter the path to the second model .bin file (model2.bin): ")

    # Load the configuration for the model you are working with
    config = AutoConfig.from_pretrained(config_path)

    # Load the two separate model .bin files
    model_1 = torch.load(model1_path)
    model_2 = torch.load(model2_path)

    # Merge the state dictionaries of the two models
    merged_state_dict = model_1.copy()
    for key, value in model_2.items():
        if key not in merged_state_dict:
            merged_state_dict[key] = value
        else:
            # If the keys already exist in the first model, average the values
            merged_state_dict[key] = (merged_state_dict[key] + value) / 2

    # Save the merged model
    torch.save(merged_state_dict, "pytorch_model.bin")

    print("Merged model saved as pytorch_model.bin . Exiting.")
