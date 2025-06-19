import pkgutil
import importlib
import avalanche

def print_submodules(package, indent=0):
    """Print all submodules of a package recursively."""
    prefix = package.__name__ + '.'
    for _, name, is_pkg in pkgutil.iter_modules(package.__path__, prefix):
        print('  ' * indent + name)
        if is_pkg:
            try:
                module = importlib.import_module(name)
                print_submodules(module, indent + 1)
            except ImportError as e:
                print('  ' * (indent + 1) + f"Error importing {name}: {e}")

def main():
    print("Avalanche Package Structure:")
    print("=" * 50)
    print_submodules(avalanche)
    print("=" * 50)

if __name__ == "__main__":
    main() 