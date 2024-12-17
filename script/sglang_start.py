from sglang.utils import execute_shell_command, wait_for_server
# wait_for_server, terminate_process, print_highlight
import yaml
import argparse


def load_yaml(config_filepath):
    # Load the YAML file
    with open(config_filepath, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main():

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--cfg_path', type=str, default='../config/config.yaml', help='Path to the configuration file.')
    parser.add_argument('--model_name', type=str, default='paraphrase_model', help='Path to the configuration file.')
    parser.add_argument('--port', type=int, default=33000, help='port.')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='host.')
    parser.add_argument('--mem_fraction_static', type=float, default=0.6, help='mem_fraction_static.')

    args = parser.parse_args()

    config = load_yaml(args.cfg_path)

    server_process = execute_shell_command(
    """
    python -m sglang.launch_server \
    --model-path {} \
    --port {} \
    --host {} \
    --mem-fraction-static {}
    """.format(
        config[args.model_name]['model_path'],
        args.port,
        args.host,
        args.mem_fraction_static
    )
    )
    print(wait_for_server("http://localhost:{}".format(args.port)))


if __name__ == "__main__":
    main()