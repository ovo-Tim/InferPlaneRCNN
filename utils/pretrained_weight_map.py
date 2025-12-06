import re


def get_map_dict(_my_keys_full: list[str], _old_keys: list[str]):
    # 1. Map FPN Top-down blocks
    # My idx -> Original P-level
    fpn_level_map = {0: "P2", 1: "P3", 2: "P4", 3: "P5"}

    # 2. Map ResNet Backbone (bottom_up)
    # My idx -> Original C-level
    backbone_map = {0: "C1", 1: "C2", 2: "C3", 3: "C4", 4: "C5"}

    old_keys = list(_old_keys)
    my_keys_full = set(_my_keys_full)

    # ---------------------------------------------------------
    # PROCESSING LOGIC
    # ---------------------------------------------------------

    # We will build the dictionary by iterating over your keys
    # and applying string transformations.

    final_mapping = {}

    for my_key in my_keys_full:
        old_key = my_key  # Default to same (unlikely to match)

        # ---------------------------
        # 1. Depth Encoder
        # pattern: depth.convs.0 -> depth.conv1
        # ---------------------------
        if "depth.convs." in my_key:
            # Extract index
            match = re.search(r"depth\.convs\.(\d+)\.(.*)", my_key)
            if match:
                idx = int(match.group(1))
                rest = match.group(2)
                # old index is my index + 1
                old_key = f"depth.conv{idx + 1}.{rest}"
                final_mapping[my_key] = old_key

        # ---------------------------
        # 2. Depth Decoder
        # pattern: depth.deconvs.0 -> depth.deconv1
        # ---------------------------
        elif "depth.deconvs." in my_key:
            match = re.search(r"depth\.deconvs\.(\d+)\.(.*)", my_key)
            if match:
                idx = int(match.group(1))
                rest = match.group(2)
                old_key = f"depth.deconv{idx + 1}.{rest}"
                final_mapping[my_key] = old_key

        # ---------------------------
        # 3. FPN Top-down
        # pattern: topdown_blocks.0.0 -> P5_conv1
        # pattern: topdown_blocks.0.1 -> P5_conv2.1 (weird suffix in original)
        # ---------------------------
        elif "fpn.lateral_convs." in my_key:
            match = re.search(r"fpn\.lateral_convs\.(\d+)\.(.*)", my_key)
            if match:
                block_idx = int(match.group(1))
                rest = match.group(2)

                p_level = fpn_level_map[block_idx]  # e.g. P5

                if rest == "weight":
                    # lateral conv (1x1)
                    old_key = f"fpn.{p_level}_conv1.{rest}"
                elif rest == "bias":
                    old_key = f"fpn.{p_level}_conv1.{rest}"
                else:
                    print("Unknown rest:", rest)

                final_mapping[my_key] = old_key

        elif "fpn.fpn_convs." in my_key:
            match = re.search(r"fpn\.fpn_convs\.(\d+)\.(.*)", my_key)
            if match:
                block_idx = int(match.group(1))
                rest = match.group(2)

                p_level = fpn_level_map[block_idx]  # e.g. P5

                if rest == "weight":
                    # output conv (3x3) - Original has .1 suffix
                    old_key = f"fpn.{p_level}_conv2.1.{rest}"
                elif rest == "bias":
                    # output conv (3x3) - Original has .1 suffix
                    old_key = f"fpn.{p_level}_conv2.1.{rest}"
                else:
                    print("Unknown rest:", rest)

                final_mapping[my_key] = old_key

        # ---------------------------
        # 4. ResNet Backbone
        # pattern: bottom_up_blocks.0 -> C1
        # pattern: bottom_up_blocks.1 -> C2
        # ---------------------------
        elif "fpn.bottom_up_blocks." in my_key:
            match = re.search(r"fpn\.bottom_up_blocks\.(\d+)\.(.*)", my_key)
            if match:
                block_idx = int(match.group(1))
                rest = match.group(2)

                c_level = backbone_map[block_idx]  # e.g. C2

                old_key = f"fpn.{c_level}.{rest}"
                final_mapping[my_key] = old_key

        # ---------------------------
        # 5. Anchors (Buffers)
        # ---------------------------
        elif "anchors" in my_key:
            # Usually anchors are not loaded from state dict but generated
            # Skip or set to None
            continue
        elif my_key in old_keys:
            final_mapping[my_key] = my_key

        if old_key in old_keys:
            old_keys.remove(old_key)
        else:
            print(f"Unmatched key: {my_key} -> {old_key}")

    if old_keys:
        print(f"Unmatched keys: {old_keys}")
    return final_mapping


def load_mapped_weights(model, old_state_dict, mapping_dict):
    new_state_dict = model.state_dict()

    # Filtered dict to load
    mapped_state_dict = {}

    for my_key, old_key in mapping_dict.items():
        if old_key in old_state_dict:
            # Check shape compatibility
            if new_state_dict[my_key].shape == old_state_dict[old_key].shape:
                mapped_state_dict[my_key] = old_state_dict[old_key]
            else:
                print(
                    f"Shape mismatch for {my_key}: My {new_state_dict[my_key].shape} vs Old {old_state_dict[old_key].shape}"
                )
        else:
            # Logic to handle missing keys (specifically ResNet Stem C1)
            # The original C1 might only have bias saved or be named differently
            # if it was frozen during training.
            print(f"Missing key: {my_key} -> {old_key}")

    # Load
    model.load_state_dict(mapped_state_dict, strict=True)
    print("Weights loaded with mapping.")
