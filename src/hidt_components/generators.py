from collections import defaultdict

import torch
from torch.nn import Module

from .blocks import Conv2dBlock, ResBlocks
from .utils import weights_init, get_total_data_dim, split_tensor_to_maps


class GeneratorBase(Module):
    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self._check_config(self.params)
        self._init_modules(self.params['modules'])
        self.apply(weights_init(self.params['initialization']))

    def _init_modules(self, params):
        for module_name, module_config in params.items():
            module_config_copy = copy.deepcopy(module_config)
            architecture = module_config_copy.pop('architecture')
            frozen = module_config_copy.pop('frozen', False)

            if 'input_data' in module_config_copy:
                module_config_copy['input_dim'] = get_total_data_dim(module_config_copy['input_data'])
                module_config_copy.pop('input_data')
            if 'output_data' in module_config_copy:
                module_config_copy['output_dim'] = get_total_data_dim(module_config_copy['output_data'])
                module_config_copy.pop('output_data')

            logger.debug(f'Building {module_name} with {architecture}')
            setattr(self,
                    module_name,
                    getattr(gen_parts, architecture)(**module_config_copy)
                    )

            if frozen:
                for param in getattr(self, module_name).parameters():
                    param.requires_grad = False
                logger.debug(f'{module_name} was frozen')

    def _check_config(self, params):
        """
        Assure module has all necessary submodules with correct names
        """
        raise NotImplementedError

    def forward(self, data, mode=None):
        if mode is None:
            # reconstruct an image
            decomposition = self.encode(data)
            output = self.decode(decomposition)
            return output
        if mode == 'encode':
            return self.encode(data)
        if mode == 'decode':
            return self.decode(data)
        if mode == 'mapper':
            return self.mapper(data['style'])

    def encode(self, data):
        """
        Define input tensors for encoders here and process those tensors
        """
        raise NotImplementedError

    def decode(self, decomposition):
        """
        Take necessary tensors from the given decomposition and pass through your decoder
        """
        raise NotImplementedError

    def _split_decoded_tensor_to_maps(self, tensor) -> dict:
        return split_tensor_to_maps(tensor, self.params['modules']['decoder']['output_data'])


class GeneratorContentStyle(GeneratorBase):
    # AdaIN auto-encoder architecture

    def __init__(self, params):
        super().__init__(params)
        self.style_dim = self.decoder.style_dim

    def _check_config(self, params):
        assert 'content_encoder' in params['modules']
        assert 'style_encoder' in params['modules']
        assert 'decoder' in params['modules']

    def encode_style(self, data, batch_size=None):
        styles = []
        styles.append(self.style_encoder(data))

        return dict(
            style=torch.cat(styles),
        )

    def encode_style_batch(self, data, batch_size=None):
        styles = []
        if batch_size is None:
            batch_size = data['images'].shape[0]

        for images in data['images'].split(batch_size):
            styles.append(self.style_encoder(images))

        return dict(
            style=torch.cat(styles),
        )

    def encode_content(self, data, batch_size=None):
        contents = []
        if batch_size is None:
            batch_size = data['images'].shape[0]

        for images in data['images'].split(batch_size):
            contents.append(self.content_encoder(images))

        return dict(
            content=torch.cat(contents),
        )

    def encode(self, data, batch_size=None):
        output = self.encode_content(data, batch_size=batch_size)
        output.update(self.encode_style(data, batch_size=batch_size))
        return output

    def decode(self, decomposition, batch_size=None):
        if batch_size is None:
            batch_size = decomposition['content'].shape[0]
        output_maps = defaultdict(list)

        for cur_content, cur_style in zip(decomposition['content'].split(batch_size),
                                          decomposition['style'].split(batch_size)):
            cur_tensor = self.decoder(cur_content, cur_style)
            cur_maps = self._split_decoded_tensor_to_maps(cur_tensor)
            for map_name, map_value in cur_maps.items():
                output_maps[map_name].append(map_value)

        output_maps = {map_name: torch.cat(map_value) for map_name, map_value in output_maps.items()}
        return output_maps


class GeneratorContentStyleUnet(GeneratorContentStyle):
    def encode_content(self, data, batch_size=None):
        contents = []
        intermediate_outputs = []

        content_outputs = self.content_encoder(data)
        contents.append(content_outputs[0])
        intermediate_outputs.append(content_outputs[1:])

        return dict(
            content=torch.cat(contents),
            intermediate_outputs=[torch.cat(out) for out in zip(*intermediate_outputs)]
        )

    def encode_content_batch(self, data, batch_size=None):
        contents = []
        intermediate_outputs = []
        if batch_size is None:
            batch_size = data['images'].shape[0]

        if isinstance(batch_size, torch.TensorType):
            batch_size = batch_size.item()

        for images in data['images'].split(int(batch_size)):
            content_outputs = self.content_encoder(images)
            contents.append(content_outputs[0])
            intermediate_outputs.append(content_outputs[1:])

        return dict(
            content=torch.cat(contents),
            intermediate_outputs=[torch.cat(out) for out in zip(*intermediate_outputs)]
        )

    def decode(self, decomposition, batch_size=None, pure_generation=False):
        if batch_size is None:
            batch_size = decomposition['content'].shape[0]
        output_maps = defaultdict(list)

        cur_content_inputs = [decomposition['content']] + list(decomposition['intermediate_outputs'])
        cur_tensor = self.decoder(cur_content_inputs, decomposition['style'], pure_generation=pure_generation)
        cur_maps = self._split_decoded_tensor_to_maps(cur_tensor)
        for map_name, map_value in cur_maps.items():
            output_maps[map_name].append(map_value)

        output_maps = {map_name: torch.cat(map_value) for map_name, map_value in output_maps.items()}
        return output_maps