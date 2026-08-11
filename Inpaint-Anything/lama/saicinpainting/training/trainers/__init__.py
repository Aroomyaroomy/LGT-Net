"""Minimal trainers for inference — skips heavy training deps during build verification."""

def load_checkpoint(train_config, path, map_location='cuda', strict=True):
    """Load a pretrained LaMa model checkpoint (lazy import to avoid build-time cascade)."""
    import torch
    from saicinpainting.training.trainers.default import DefaultInpaintingTrainingModule

    def _make_model(config):
        kind = config.training_model.kind
        if kind != 'default':
            raise ValueError(f'Unknown trainer module {kind}')
        kwargs = dict(config.training_model)
        kwargs.pop('kind')
        kwargs['use_ddp'] = config.trainer.kwargs.get('accelerator', None) == 'ddp'
        return DefaultInpaintingTrainingModule(config, **kwargs)

    model: torch.nn.Module = _make_model(train_config)
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state['state_dict'], strict=strict)
    model.on_load_checkpoint(state)
    return model
