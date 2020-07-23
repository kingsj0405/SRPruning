# Pytorch-Based method

def prune_with_synflow(model, origin_channel, prune_channel_list, filter_unprune, filter_in, filter_out):
    # Copy model
    model_new = copy.deepcopy(model)
    model_new.train()
    for prune_channel in prune_channel_list:
        # Variable
        amount = prune_channel / origin_channel
        logger.debug("=" * 20)
        logger.debug("=" * 20)
        logger.debug(f"{origin_channel} to {origin_channel - prune_channel} with amount: {amount}")
        logger.debug("=" * 20)
        logger.debug("=" * 20)
        origin_channel -= prune_channel
        # Get absolute model
        model_abs = copy.deepcopy(model_new)
        for name, module in model_abs.named_modules():
            # absolute or not
            module_name = type(module).__name__
            unabsolute = 'Conv2d' not in module_name
            if unabsolute: continue
            # absolute
            module.weight = torch.nn.Parameter(module.weight.abs() * 1e-2)
            module.bias = torch.nn.Parameter(torch.zeros_like(module.bias))
        # SynFlow
        model_abs.zero_grad()
        img = torch.ones((1, 3, 32, 32))
        out = model_abs(img)
        loss = out.sum()
        loss.backward()
        # do pruning
        for name, module in model_new.named_modules():
            # Prune or not
            module_name = type(module).__name__
            unprune = 'Conv2d' not in module_name or name in filter_unprune
            logger.debug("=" * 20)
            logger.debug(f"{name}, {module_name}: Prune-{not unprune}")
            if unprune: continue
            # Get absolute module with gradient
            module_abs = None
            for n, m in model_abs.named_modules():
                if name == n:
                    module_abs = m
                    break
            # Prepare pruning
            pre_module_weight_shape = module.weight.shape
            pre_module_bias_shape = module.bias.shape
            # SynFlow score
            module_abs.weight = torch.nn.Parameter(torch.mul(module_abs.weight, module_abs.weight.grad))
            # Prune in_channel
            if name not in filter_in:
                prune.ln_structured(module_abs, 'weight', amount=amount, n=2, dim=1)
                prune.remove(module_abs, 'weight')
                in_channel_mask = module_abs.weight.sum(-1).sum(-1).sum(0) != 0
            else:
                in_channel_mask = torch.ones(module.weight.shape[1], dtype=torch.bool).to(device)
            # Prune out_channel
            if name not in filter_out:
                prune.ln_structured(module_abs, 'weight', amount=amount, n=2, dim=0)
                prune.remove(module_abs, 'weight')
                out_channel_mask = module_abs.weight.sum(-1).sum(-1).sum(-1) != 0
            else:
                out_channel_mask = torch.ones(module.weight.shape[0], dtype=torch.bool).to(device)
            module.weight = torch.nn.Parameter(module.weight[:,in_channel_mask,:,:])
            module.weight = torch.nn.Parameter(module.weight[out_channel_mask,:,:,:])
            module.bias = torch.nn.Parameter(module.bias[out_channel_mask])
            logger.debug(f"mask_index.shape: {out_channel_mask.shape}")
            logger.debug(f"module.weight.shape: {pre_module_weight_shape} --> {module.weight.shape}")
            logger.debug(f"module.bias.shape: {pre_module_bias_shape} --> {module.bias.shape}")
    return model_new


def prune_with_srflow(model, origin_channel, prune_channel_list, filter_unprune, filter_in, filter_out):
    # Copy model
    model_new = copy.deepcopy(model)
    model_new.train()
    for prune_channel in prune_channel_list:
        # Variable
        amount = prune_channel / origin_channel
        logger.debug("=" * 20)
        logger.debug("=" * 20)
        logger.debug(f"{origin_channel} to {origin_channel - prune_channel} with amount: {amount}")
        logger.debug("=" * 20)
        logger.debug("=" * 20)
        origin_channel -= prune_channel
        # SRFlow
        model_flow = copy.deepcopy(model_new)
        model_flow.zero_grad()
        img_lr = torch.ones((1, 3, 32, 32))
        img_hr = torch.ones((1, 3, 32 * 4, 32 * 4))
        img_sr = model_flow(img_lr)
        loss = torch.nn.MSELoss()(img_hr, img_sr)
        loss.backward()
        # do pruning
        for name, module in model_new.named_modules():
            # Prune or not
            module_name = type(module).__name__
            unprune = 'Conv2d' not in module_name or name in filter_unprune
            logger.debug("=" * 20)
            logger.debug(f"{name}, {module_name}: Prune-{not unprune}")
            if unprune: continue
            # Get flow module with gradient
            module_flow = None
            for n, m in model_flow.named_modules():
                if name == n:
                    module_flow = m
                    break
            # Prepare pruning
            pre_module_weight_shape = module.weight.shape
            pre_module_bias_shape = module.bias.shape
            # SRFlow score
            module_flow.weight = torch.nn.Parameter(torch.mul(module_flow.weight.abs(), module_flow.weight.grad.abs()))
            # Prune in_channel
            if name not in filter_in:
                prune.ln_structured(module_flow, 'weight', amount=amount, n=2, dim=1)
                prune.remove(module_flow, 'weight')
                in_channel_mask = module_flow.weight.sum(-1).sum(-1).sum(0) != 0
            else:
                in_channel_mask = torch.ones(module.weight.shape[1], dtype=torch.bool).to(device)
            # Prune out_channel
            if name not in filter_out:
                prune.ln_structured(module_flow, 'weight', amount=amount, n=2, dim=0)
                prune.remove(module_flow, 'weight')
                out_channel_mask = module_flow.weight.sum(-1).sum(-1).sum(-1) != 0
            else:
                out_channel_mask = torch.ones(module.weight.shape[0], dtype=torch.bool).to(device)
            module.weight = torch.nn.Parameter(module.weight[:,in_channel_mask,:,:])
            module.weight = torch.nn.Parameter(module.weight[out_channel_mask,:,:,:])
            module.bias = torch.nn.Parameter(module.bias[out_channel_mask])
            logger.debug(f"mask_index.shape: {out_channel_mask.shape}")
            logger.debug(f"module.weight.shape: {pre_module_weight_shape} --> {module.weight.shape}")
            logger.debug(f"module.bias.shape: {pre_module_bias_shape} --> {module.bias.shape}")
    return model_new


def prune_with_srflow2(model, origin_channel, prune_channel_list, filter_unprune, filter_in, filter_out, device, epoch=50):
    # Copy model
    model_new = copy.deepcopy(model)
    model_new.train()
    # Image
    img_lr = torch.ones((1, 3, 32, 32)).to(device)
    img_hr = torch.ones((1, 3, 32 * 4, 32 * 4)).to(device)
    criterion = torch.nn.MSELoss()
    for i, prune_channel in enumerate(prune_channel_list):
        # Variable
        amount = prune_channel / origin_channel
        logger.debug("=" * 20)
        logger.debug("=" * 20)
        logger.debug(f"{origin_channel} to {origin_channel - prune_channel} with amount: {amount}")
        logger.debug("=" * 20)
        logger.debug("=" * 20)
        origin_channel -= prune_channel
        # SRFlow 2
        model_flow = copy.deepcopy(model_new)
        ## Train model a little bit
        optimizer = torch.optim.Adam(model_flow.parameters())
        for j in range(epoch):
            optimizer.zero_grad()
            img_sr = model_flow(img_lr)
            loss = criterion(img_hr, img_sr)
            loss.backward()
            optimizer.step()
            logger.debug(f"[{i + 1}/{len(prune_channel_list)}][{j + 1}/{epoch}] loss: {loss}")
        ## Calculate SRFlow2 Score
        model_flow.zero_grad()
        img_sr = model_flow(img_lr)
        loss = criterion(img_hr, img_sr)
        loss.backward()
        # do pruning
        for name, module in model_new.named_modules():
            # Prune or not
            module_name = type(module).__name__
            unprune = 'Conv2d' not in module_name or name in filter_unprune
            logger.debug("=" * 20)
            logger.debug(f"{name}, {module_name}: Prune-{not unprune}")
            if unprune: continue
            # Get flow module with gradient
            module_flow = None
            for n, m in model_flow.named_modules():
                if name == n:
                    module_flow = m
                    break
            # Prepare pruning
            pre_module_weight_shape = module.weight.shape
            pre_module_bias_shape = module.bias.shape
            # SRFlow score
            module_flow.weight = torch.nn.Parameter(torch.mul(module_flow.weight.abs(), module_flow.weight.grad.abs()))
            # Prune in_channel
            if name not in filter_in:
                prune.ln_structured(module_flow, 'weight', amount=amount, n=2, dim=1)
                prune.remove(module_flow, 'weight')
                in_channel_mask = module_flow.weight.sum(-1).sum(-1).sum(0) != 0
            else:
                in_channel_mask = torch.ones(module.weight.shape[1], dtype=torch.bool).to(device)
            # Prune out_channel
            if name not in filter_out:
                prune.ln_structured(module_flow, 'weight', amount=amount, n=2, dim=0)
                prune.remove(module_flow, 'weight')
                out_channel_mask = module_flow.weight.sum(-1).sum(-1).sum(-1) != 0
            else:
                out_channel_mask = torch.ones(module.weight.shape[0], dtype=torch.bool).to(device)
            module.weight = torch.nn.Parameter(module.weight[:,in_channel_mask,:,:])
            module.weight = torch.nn.Parameter(module.weight[out_channel_mask,:,:,:])
            module.bias = torch.nn.Parameter(module.bias[out_channel_mask])
            logger.debug(f"mask_index.shape: {out_channel_mask.shape}")
            logger.debug(f"module.weight.shape: {pre_module_weight_shape} --> {module.weight.shape}")
            logger.debug(f"module.bias.shape: {pre_module_bias_shape} --> {module.bias.shape}")
    return model_new


def prune_with_random(model, origin_channel, prune_channel_list, filter_unprune, filter_in, filter_out, device):
    # Copy model
    model_new = copy.deepcopy(model)
    model_new.train()
    for prune_channel in prune_channel_list:
        # Variable
        amount = prune_channel / origin_channel
        logger.debug("=" * 20)
        logger.debug("=" * 20)
        logger.debug(f"{origin_channel} to {origin_channel - prune_channel} with amount: {amount}")
        logger.debug("=" * 20)
        logger.debug("=" * 20)
        origin_channel -= prune_channel
        # do pruning
        for name, module in model_new.named_modules():
            # Prune or not
            module_name = type(module).__name__
            unprune = 'Conv2d' not in module_name or name in filter_unprune
            logger.debug("=" * 20)
            logger.debug(f"{name}, {module_name}: Prune-{not unprune}")
            if unprune: continue
            # Prepare pruning
            pre_module_weight_shape = module.weight.shape
            pre_module_bias_shape = module.bias.shape
            # Prune in_channel
            if name not in filter_in:
                prune.random_structured(module, 'weight', amount=amount, dim=1)
                prune.remove(module, 'weight')
                in_channel_mask = module.weight.sum(-1).sum(-1).sum(0) != 0
            else:
                in_channel_mask = torch.ones(module.weight.shape[1], dtype=torch.bool).to(device)
            # Prune out_channel
            if name not in filter_out:
                prune.random_structured(module, 'weight', amount=amount, dim=0)
                prune.remove(module, 'weight')
                out_channel_mask = module.weight.sum(-1).sum(-1).sum(-1) != 0
            else:
                out_channel_mask = torch.ones(module.weight.shape[0], dtype=torch.bool).to(device)
            module.weight = torch.nn.Parameter(module.weight[:,in_channel_mask,:,:])
            module.weight = torch.nn.Parameter(module.weight[out_channel_mask,:,:,:])
            module.bias = torch.nn.Parameter(module.bias[out_channel_mask])
            logger.debug(f"mask_index.shape: {out_channel_mask.shape}")
            logger.debug(f"module.weight.shape: {pre_module_weight_shape} --> {module.weight.shape}")
            logger.debug(f"module.bias.shape: {pre_module_bias_shape} --> {module.bias.shape}")
    return model_new