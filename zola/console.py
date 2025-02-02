from rich.console import Console
from rich.traceback import install

console = Console()

# Install rich traceback handling
install(show_locals=True)
