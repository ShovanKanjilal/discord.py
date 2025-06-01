import sys
import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Path to the bot.py file
bot_file = 'bot.py'

# Function to run the bot
def run_bot():
    return subprocess.Popen([sys.executable, bot_file])

# Event handler class to monitor changes in bot.py
class BotFileChangeHandler(FileSystemEventHandler):
    def __init__(self, process):
        self.process = process

    def on_modified(self, event):
        if event.src_path.endswith(bot_file):
            print(f'{bot_file} has been modified. Restarting the bot...')
            self.process.terminate()  # Terminate the current bot process
            self.process.wait()  # Wait for the bot process to terminate cleanly
            self.process = run_bot()  # Restart the bot

    def on_created(self, event):
        if event.src_path.endswith(bot_file):
            print(f'{bot_file} has been created. Starting the bot...')
            self.process = run_bot()

if __name__ == "__main__":
    # Initially run the bot
    process = run_bot()
    
    # Set up a watchdog observer to monitor changes in the directory
    event_handler = BotFileChangeHandler(process)
    observer = Observer()
    observer.schedule(event_handler, path='.', recursive=False)
    
    print(f"Watching for changes in {bot_file}...")
    
    # Start the observer
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        process.terminate()  # Cleanly stop the bot on exit
    observer.join()
