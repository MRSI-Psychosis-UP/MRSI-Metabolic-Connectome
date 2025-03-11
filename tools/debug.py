from rich.console import Console
from rich.markup import escape
from rich.theme import Theme
from rich.progress import Progress
import threading, time

class Debug:
    def __init__(self, verbose=True):
        custom_theme = Theme({
            "success": "green",
            "error": "red",
            "warning": "yellow",
            "info": "blue",
            "failure": "bold red",
            "debug": "magenta"
        })
        self.console = Console(theme=custom_theme)
        self.verbose = verbose

    def concatenate(self, *messages):
        return ' '.join(str(message) for message in messages)

    def success(self, *messages):
        if self.verbose:
            message = self.concatenate(*messages)
            self.console.print(f"[success][ SUCCESS ][/success] {escape(message)}")

    def error(self, *messages):
        if self.verbose:
            message = self.concatenate(*messages)
            self.console.print(f"[error][  ERROR  ][/error] {escape(message)}")

    def warning(self, *messages):
        if self.verbose:
            message = self.concatenate(*messages)
            self.console.print(f"[warning][ WARNING ][/warning] {escape(message)}")

    def failure(self, *messages):
        if self.verbose:
            message = self.concatenate(*messages)
            self.console.print(f"[failure][ FAILURE ][/failure] {escape(message)}")

    def info(self, *messages):
        if self.verbose:
            message = self.concatenate(*messages)
            self.console.print(f"[info][   INFO  ][/info] {escape(message)}")

    def separator(self):
        if self.verbose:
            self.console.rule()

    def title(self, title):
        if self.verbose:
            self.console.rule(title, style="debug")

    def __progress_bar_task(self,duration):
        with Progress() as progress:
            # Create a task with a total of `duration` seconds
            task = progress.add_task("[green]Registration...", total=duration)
            
            # Update the progress bar every second
            for _ in range(duration):
                time.sleep(1)  # Wait for a second
                progress.update(task, advance=1)  # Advance the progress bar by 1

    def run_progress_bar_in_background(self,duration):
        # Create a thread that runs the progress bar task
        thread = threading.Thread(target=self.__progress_bar_task, args=(duration,),daemon=True)
        thread.start()


if __name__=="__main__":
    debug = Debug()
    value = 23
    message="is the question to the ultimate answer of the Universe."
    debug.separator()
    debug.title("THIS IS A TEST")
    debug.success(value,message)
    debug.error(value,message)
    debug.warning(value,message)
    debug.failure(value,message)
    debug.info(value,message)
    debug.separator()