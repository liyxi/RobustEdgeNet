import yaml

from argparse import ArgumentParser, ArgumentError


def get_flat_args(config):
    args = {}
    for k in config:
        if isinstance(config[k], dict):
            v = get_flat_args(config[k])
            for k2 in v:
                args['_'.join([k, k2])] = v[k2]
        else:
            args[k] = config[k]
    return args


class Config(object):

    def __init__(self, default_config_file=''):
        super(Config, self).__init__()
        self.config_parser = ArgumentParser(description='Training Config', add_help=False)
        self.config_parser.add_argument('-c', '--config', default=default_config_file, type=str, metavar='FILE',
                                        help='YAML config file specifying default arguments')

        self.parser = ArgumentParser(description='Training Config')

    def load_config(self, args=None):

        if args is not None:
            config_args, remaining_args = self.config_parser.parse_known_args(args)
        else:
            config_args, remaining_args = self.config_parser.parse_known_args()

        # load from file
        if len(config_args.config) > 0:
            with open(config_args.config, 'r', encoding='utf-8') as fp:
                config_yaml = yaml.safe_load(fp.read())
            # get flat args
            yaml_args = get_flat_args(config_yaml)
        else:
            yaml_args = {}

        for k in yaml_args:
            v = yaml_args[k]
            try:
                if isinstance(v, bool):
                    self.parser.add_argument('--%s' % (k,), default=v, action='store_true')
                elif isinstance(v, (list, tuple)):
                    self.parser.add_argument('--%s' % (k,), default=v, nargs='*')
                else:
                    self.parser.add_argument('--%s' % (k,), type=type(v), default=v)
            except ArgumentError:
                self.parser.set_defaults(**{k: v})

        args, remaining_args = self.parser.parse_known_args(remaining_args)
        if len(remaining_args) > 0:
            import warnings
            warnings.warn(str(remaining_args))
        return args
