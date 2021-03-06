import click

import train, infer

@click.group()
def main():
    pass


def add_module_to_group(module, group):
    for name in module.__all__:
        group.add_command(getattr(module, name))


add_module_to_group(train, main)
add_module_to_group(infer, main)


if __name__ == "__main__":
    main()

