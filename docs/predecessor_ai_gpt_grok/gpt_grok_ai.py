import os
import json
import openai
import requests
import subprocess
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
import time
import logging
from threading import Thread
from queue import Queue
import yaml
import sys
import traceback
import shutil
import importlib
from io import StringIO

# Configuration
CONFIG_FILE = "config.yaml"
MEMORY_FILE = "ai_memory.json"
LOG_FILE = "ai_log.txt"
SCRIPT_FILE = "gpt_grok_ai.py"
BACKUP_DIR = "backups"
TASK_DIR = "tasks"

# Setup logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ensure directories exist
for directory in [BACKUP_DIR, TASK_DIR]:
    os.makedirs(directory, exist_ok=True)

# Custom stream redirector to capture terminal output
class StreamRedirector(StringIO):
    def __init__(self, update_status_callback=None):
        super().__init__()
        self.update_status_callback = update_status_callback
        self.buffer = ""

    def write(self, text):
        self.buffer += text
        if self.update_status_callback:
            self.update_status_callback(f"Stream Output: {text.strip()}", 'info')

    def flush(self):
        pass

    def get_output(self):
        return self.buffer

# Load or initialize config
def load_config():
    print("Loading config...")
    defaults = {
        "openai_api_key": "",
        "grok_api_key": "",
        "claude_api_key": "",
        "google_api_key": "",
        "search_engine_id": ""
    }
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                config = yaml.safe_load(f)
                print(f"Config loaded: {config}")
                return {**defaults, **config}
        else:
            print("Config file not found, creating default...")
            with open(CONFIG_FILE, "w") as f:
                yaml.safe_dump(defaults, f)
            return defaults
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        return defaults

CONFIG = load_config()
openai_client = openai.OpenAI(api_key=CONFIG["openai_api_key"]) if CONFIG["openai_api_key"] else None
grok_client = openai.OpenAI(api_key=CONFIG["grok_api_key"], base_url="https://api.x.ai/v1") if CONFIG["grok_api_key"] else None
claude_headers = {"x-api-key": CONFIG["claude_api_key"], "anthropic-version": "2023-06-01"} if CONFIG["claude_api_key"] else None

# Memory management
def load_memory():
    print("Loading memory...")
    try:
        if not os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "w") as f:
                json.dump({"tasks": {}}, f)
        with open(MEMORY_FILE, "r") as f:
            return json.load(f).get("tasks", {})
    except Exception as e:
        print(f"Error loading memory: {str(e)}")
        return {}

def save_memory(memory):
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump({"tasks": memory}, f, indent=4)
    except Exception as e:
        print(f"Error saving memory: {str(e)}")

# Web search for solutions (Agent 4: Google)
def web_search(query):
    if not CONFIG["google_api_key"] or not CONFIG["search_engine_id"]:
        return "Error: Missing Google API credentials.", "Error"
    try:
        url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={CONFIG['google_api_key']}&cx={CONFIG['search_engine_id']}"
        response = requests.get(url, timeout=10)
        results = response.json().get("items", [])
        return "\n".join([result["snippet"] for result in results[:3]]) if results else "No results.", "Success"
    except Exception as e:
        logging.error(f"Web search failed: {e}")
        return f"Error: {str(e)}", "Error"

# Agent 1: OpenAI - Planner
def plan_task(task):
    if not openai_client:
        return f"Error: No OpenAI API key. Using task as plan: {task}", "Error"
    prompt = f"Plan the steps to accomplish this task in Python: {task}"
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message.content, "Success"
    except Exception as e:
        return f"Error: {str(e)}. Using task as plan: {task}", "Error"

# Agent 2: Grok - Generator
def generate_code_grok(task, plan="", current_code=""):
    if not grok_client:
        return current_code, "Error: No Grok API key."
    prompt = (
        f"Task: {task}\n"
        f"Plan: {plan}\n"
        f"Current Code (start fresh if empty):\n```python\n{current_code}\n```\n"
        "Generate creative Python code for this task."
    )
    try:
        response = grok_client.chat.completions.create(
            model="grok-2-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        logging.info(f"Grok response: {response}")
        if not response or not hasattr(response, 'choices') or not response.choices:
            return current_code, "Error: Grok returned no choices."
        content = response.choices[0].message.content
        if not content:
            return current_code, "Error: Grok returned no valid content."
        if "```python" not in content:
            return content.strip(), "Success"
        new_code = content.split("```python")[1].split("```")[0].strip()
        return new_code, "Success"
    except Exception as e:
        logging.error(f"Grok API error: {str(e)}")
        return current_code, f"Error: {str(e)}"

# Agent 3: Claude - Refiner
def refine_code_claude(code, task):
    if not claude_headers:
        return code, "Error: No Claude API key."
    try:
        response = requests.post("https://api.anthropic.com/v1/messages", headers=claude_headers, json={
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{
                "role": "user",
                "content": (
                    f"Refine this Python code for task '{task}':\n```python\n{code}\n```\n"
                    "Preserve its core intent, even if unconventional, and focus on improving clarity and functionality."
                )
            }],
            "max_tokens": 1000
        })
        response.raise_for_status()
        content = response.json()["content"][0]["text"]
        refined_code = content.split("```python")[1].split("```")[0].strip() if "```python" in content else content.strip()
        return refined_code, "Success"
    except Exception as e:
        logging.error(f"Claude API error: {str(e)}")
        return code, f"Error: {str(e)}"

# Execute code and capture terminal output
def execute_code(code, timeout=10, update_status_callback=None):
    def run_with_output(q):
        try:
            temp_file = "temp_execute.py"
            with open(temp_file, "w") as f:
                f.write(code)
            process = subprocess.Popen(
                [sys.executable, temp_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate(timeout=timeout)
            os.remove(temp_file)
            if process.returncode == 0:
                q.put(("Execution successful", True, stdout + stderr))
            else:
                q.put((f"Error: {stderr}", False, stdout + stderr))
        except subprocess.TimeoutExpired:
            process.kill()
            q.put(("Error: Execution timed out", False, ""))
        except Exception as e:
            stack_trace = traceback.format_exc()
            q.put((f"Error: {str(e)}", False, stack_trace))

    q = Queue()
    thread = Thread(target=run_with_output, args=(q,))
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return "Error: Execution timed out", False, ""
    result, success, terminal_output = q.get()
    if update_status_callback and terminal_output:
        update_status_callback(f"Terminal Output: {terminal_output}", 'info')
    return result, success, terminal_output

# Check if code meets Doom 1 replica criteria
def is_doom_replica(code):
    required_features = [
        "pygame",  # Graphics engine
        "wad",     # WAD file parsing
        "raycast", # Raycasting for rendering
        "enemy",   # Enemy AI
        "weapon",  # Weapon mechanics
    ]
    code_lower = code.lower()
    return all(feature in code_lower for feature in required_features)

# Self-improving AI loop with terminal and API stream reading
def self_improving_ai_loop(task, max_tries, status_text, progress_bar, progress_label, cancel_flag, root, launch_button, task_file_var, stream_redirector, start_button, cancel_button):
    def update_status(text, tag='info'):
        status_text.insert(tk.END, text + "\n", tag)
        status_text.yview(tk.END)
        root.update_idletasks()

    memory = load_memory()
    code = memory.get(task, {}).get("code", "")
    iteration = memory.get(task, {}).get("iterations", 0)
    feedback = "Initial attempt"
    max_iter = max_tries if max_tries > 0 else float('inf')

    while iteration < max_iter and not cancel_flag[0]:
        iteration += 1
        update_status(f"Iteration {iteration}/{max_iter if max_tries else '∞'}: Starting...", 'info')

        # Agent 1: OpenAI - Plan the task
        plan, plan_status = plan_task(task)
        update_status(f"Agent 1 (OpenAI) Plan: {plan}", 'info' if plan_status == "Success" else 'error')
        initial_code = ""
        if plan_status == "Success" and "```python" in plan:
            initial_code = plan.split("```python")[1].split("```")[0].strip()
        if plan_status == "Error":
            plan = task

        # Agent 2: Grok - Generate initial code
        code, grok_status = generate_code_grok(task, plan, code or initial_code)
        update_status(f"Agent 2 (Grok) Code Generation: {grok_status}", 'info' if grok_status == "Success" else 'error')
        if grok_status == "Error" and initial_code:
            code = initial_code

        # Agent 4: Google - Research additional info
        research, research_status = web_search(task + " python solution")
        update_status(f"Agent 4 (Google) Research: {research}", 'info' if research_status == "Success" else 'error')
        if research_status == "Error":
            research = "No external research available."

        # Agent 3: Claude - Refine the code
        code_to_refine = code or initial_code
        if code_to_refine:
            code, claude_status = refine_code_claude(code_to_refine, task)
            update_status(f"Agent 3 (Claude) Refinement: {claude_status}", 'info' if claude_status == "Success" else 'error')
        else:
            update_status("Agent 3 (Claude) Refinement: Skipped due to no valid code to refine", 'error')

        # Proceed with execution and capture terminal output
        if code:
            is_self_rewrite = "improve your own code" in task.lower()
            exec_result, success, terminal_output = execute_code(code, update_status_callback=update_status)
            update_status(f"Execution: {exec_result}", 'success' if success else 'error')

            # Read stream for additional output
            stream_output = stream_redirector.get_output()
            if stream_output:
                update_status(f"Stream Output: {stream_output}", 'info')
                feedback += f"\nStream Feedback: {stream_output}"

            # Check if it's a Doom replica (only for Doom tasks)
            doom_complete = "doom" in task.lower() and is_doom_replica(code)

            if is_self_rewrite:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                backup_file = os.path.join(BACKUP_DIR, f"gpt_grok_ai_backup_{timestamp}.py")
                shutil.copy(SCRIPT_FILE, backup_file)
                with open(SCRIPT_FILE, "w") as f:
                    f.write(code)
                update_status(f"Self-rewrite complete. Backup at {backup_file}. Restarting...", 'success')
                root.destroy()
                subprocess.Popen([sys.executable, SCRIPT_FILE])
                sys.exit(0)
            elif success:  # Save any successful code as a .py file
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                task_file = os.path.join(TASK_DIR, f"task_{task.lower().replace(' ', '_')}_{timestamp}.py")
                with open(task_file, "w") as f:
                    f.write(code)
                task_file_var.set(task_file)
                launch_button.config(state="normal")
                update_status(f"Task completed and saved as {task_file}!", 'success')
                memory[task] = {"code": code, "iterations": iteration}
                save_memory(memory)
                if not doom_complete:  # Continue iterating for Doom tasks unless complete
                    feedback = f"Code runs but lacks Doom 1 replica features (WAD parsing, raycasting, enemies, weapons) if applicable.\nTerminal Output: {terminal_output}\nResearch: {research}"
                    update_status(f"Feedback for next iteration: {feedback}", 'info')
                    if openai_client:
                        combined_feedback = f"{terminal_output}\n{stream_output}\n{feedback}"
                        plan, _ = plan_task(f"Improve this code to better meet the task '{task}' based on feedback: {combined_feedback}")
                        update_status(f"Agent 1 (OpenAI) Re-plan: {plan}", 'info')
                    continue
                break
            else:
                feedback = f"Error occurred: {exec_result}\nTerminal Output: {terminal_output}\nResearch: {research}"
                update_status(f"Feedback for next iteration: {feedback}", 'info')
                if openai_client:
                    combined_feedback = f"{terminal_output}\n{stream_output}\n{feedback}"
                    plan, _ = plan_task(f"Improve this code to better meet the task '{task}' based on feedback: {combined_feedback}")
                    update_status(f"Agent 1 (OpenAI) Re-plan: {plan}", 'info')
        else:
            update_status("No valid code generated to execute.", 'error')
            feedback = f"Research: {research}"

        progress_bar["value"] = (iteration / max_iter) * 100 if max_tries else min(iteration * 10, 100)
        progress_label.config(text=f"Progress: {int(progress_bar['value'])}%")
        time.sleep(0.5)

    if cancel_flag[0]:
        update_status("Task cancelled.", 'error')
    start_button.config(state="normal")
    cancel_button.config(state="disabled")

# GUI setup
def run_gui():
    print("Starting GUI setup...")
    try:
        root = tk.Tk()
        print("Tkinter root initialized.")
    except Exception as e:
        print(f"Error initializing Tkinter: {str(e)}")
        sys.exit(1)

    root.title("Four-Agent AI System")
    root.geometry("700x600")
    root.resizable(False, False)

    style = ttk.Style()
    style.theme_use('clam')

    # Define stream_redirector in run_gui scope
    stream_redirector = StreamRedirector()

    # Define functions before using them
    def start_ai(task, max_tries):
        if not task:
            messagebox.showerror("Error", "Please enter a task.")
            return
        try:
            max_tries = int(max_tries)
        except ValueError:
            messagebox.showerror("Error", "Max tries must be an integer.")
            return
        if not any([CONFIG["openai_api_key"], CONFIG["grok_api_key"], CONFIG["claude_api_key"]]):
            messagebox.showerror("Error", "Configure at least one AI API key in Settings.")
            return
        cancel_flag[0] = False
        start_button.config(state="disabled")
        cancel_button.config(state="normal")
        launch_button.config(state="disabled")
        stream_redirector.update_status_callback = lambda text, tag='info': status_text.insert(tk.END, text + "\n", tag) and status_text.yview(tk.END)
        sys.stdout = stream_redirector
        sys.stderr = stream_redirector
        Thread(target=self_improving_ai_loop, args=(task, max_tries, status_text, progress_bar, progress_label, cancel_flag, root, launch_button, task_file_var, stream_redirector, start_button, cancel_button), daemon=True).start()

    def cancel_ai():
        cancel_flag[0] = True
        start_button.config(state="normal")
        cancel_button.config(state="disabled")
        launch_button.config(state="disabled")

    def launch_task():
        task_file = task_file_var.get()
        if task_file and os.path.exists(task_file):
            subprocess.Popen([sys.executable, task_file])
        else:
            messagebox.showerror("Error", "No task file to launch.")

    def open_settings():
        settings_win = tk.Toplevel(root)
        settings_win.title("Settings")
        settings_win.geometry("400x450")
        entries = {}
        for key, label in [
            ("openai_api_key", "OpenAI API Key:"),
            ("grok_api_key", "Grok API Key:"),
            ("claude_api_key", "Claude API Key:"),
            ("google_api_key", "Google API Key:"),
            ("search_engine_id", "Search Engine ID:")
        ]:
            ttk.Label(settings_win, text=label).pack(pady=5)
            entry = ttk.Entry(settings_win, width=40)
            entry.insert(0, CONFIG[key])
            entry.pack(pady=5)
            entries[key] = entry
        ttk.Button(settings_win, text="Save", command=lambda: [CONFIG.update({k: e.get() for k, e in entries.items()}), yaml.safe_dump(CONFIG, open(CONFIG_FILE, "w")), settings_win.destroy()]).pack(pady=10)
        ttk.Button(settings_win, text="Test All APIs", command=lambda: test_all_apis(stream_redirector)).pack(pady=5)
        ttk.Button(settings_win, text="Test OpenAI Connection", command=test_openai_connection).pack(pady=5)
        ttk.Button(settings_win, text="Test Grok Connection", command=test_grok_connection).pack(pady=5)
        ttk.Button(settings_win, text="Test Claude Connection", command=test_claude_connection).pack(pady=5)
        ttk.Button(settings_win, text="Test Google Connection", command=test_google_connection).pack(pady=5)

    def test_openai_connection():
        if not CONFIG["openai_api_key"]:
            return "OpenAI: Error - API key not set."
        try:
            openai_client = openai.OpenAI(api_key=CONFIG["openai_api_key"])
            openai_client.models.list()
            return "OpenAI: Success"
        except Exception as e:
            return f"OpenAI: Error - {str(e)}"

    def test_grok_connection():
        if not CONFIG["grok_api_key"]:
            return "Grok: Error - API key not set."
        try:
            grok_client = openai.OpenAI(api_key=CONFIG["grok_api_key"], base_url="https://api.x.ai/v1")
            response = grok_client.chat.completions.create(model="grok-2-latest", messages=[{"role": "user", "content": "Test"}])
            logging.info(f"Grok test response: {response}")
            return "Grok: Success"
        except Exception as e:
            return f"Grok: Error - {str(e)}"

    def test_claude_connection():
        if not CONFIG["claude_api_key"]:
            return "Claude: Error - API key not set."
        try:
            response = requests.post("https://api.anthropic.com/v1/messages", headers=claude_headers, json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": 10
            })
            response.raise_for_status()
            return "Claude: Success"
        except Exception as e:
            return f"Claude: Error - {str(e)}"

    def test_google_connection():
        if not CONFIG["google_api_key"] or not CONFIG["search_engine_id"]:
            return "Google: Error - API key or Search Engine ID not set."
        try:
            result, status = web_search("test query")
            if status == "Error":
                return f"Google: Error - {result}"
            return "Google: Success"
        except Exception as e:
            return f"Google: Error - {str(e)}"

    def test_all_apis(stream_redirector):
        results = [
            test_openai_connection(),
            test_grok_connection(),
            test_claude_connection(),
            test_google_connection()
        ]
        print("\n".join(results))
        return "\n".join(results)

    # GUI setup continues after function definitions
    input_frame = ttk.Frame(root, padding=10)
    input_frame.pack(fill='x')

    feedback_frame = ttk.Frame(root, padding=10)
    feedback_frame.pack(fill='both', expand=True)

    button_frame = ttk.Frame(root, padding=10)
    button_frame.pack(fill='x')

    ttk.Label(input_frame, text="Task:").grid(row=0, column=0, sticky='w')
    task_entry = ttk.Entry(input_frame, width=50)
    task_entry.grid(row=0, column=1, padx=5)

    ttk.Label(input_frame, text="Max Tries (0 for infinite):").grid(row=1, column=0, sticky='w')
    max_tries_var = tk.StringVar(value="100")
    max_tries_entry = ttk.Entry(input_frame, textvariable=max_tries_var, width=10)
    max_tries_entry.grid(row=1, column=1, padx=5, sticky='w')

    status_text = scrolledtext.ScrolledText(feedback_frame, width=80, height=20, font=("Consolas", 10))
    status_text.pack(fill='both', expand=True)
    status_text.tag_config('success', foreground='green')
    status_text.tag_config('error', foreground='red')
    status_text.tag_config('info', foreground='blue')

    progress_frame = ttk.Frame(feedback_frame)
    progress_frame.pack(fill='x', pady=5)
    progress_label = ttk.Label(progress_frame, text="Progress: 0%")
    progress_label.pack(side='left')
    progress_bar = ttk.Progressbar(progress_frame, length=400, mode="determinate")
    progress_bar.pack(side='left', padx=5)

    start_button = ttk.Button(button_frame, text="Start AI", command=lambda: start_ai(task_entry.get(), max_tries_var.get()))
    start_button.pack(side='left', padx=5)

    cancel_button = ttk.Button(button_frame, text="Cancel", command=cancel_ai, state="disabled")
    cancel_button.pack(side='left', padx=5)

    launch_button = ttk.Button(button_frame, text="Launch Task", command=launch_task, state="disabled")
    launch_button.pack(side='left', padx=5)

    settings_button = ttk.Button(button_frame, text="Settings", command=open_settings)
    settings_button.pack(side='left', padx=5)

    menu_bar = tk.Menu(root)
    root.config(menu=menu_bar)
    help_menu = tk.Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="Help", menu=help_menu)
    help_menu.add_command(label="Instructions", command=lambda: print("Instructions: 1. Enter a task.\n2. Set max tries.\n3. Click 'Start AI' or Ctrl+S.\n4. Monitor progress.\n5. Launch completed tasks."))

    root.bind('<Control-s>', lambda event: start_ai(task_entry.get(), max_tries_var.get()))
    root.bind('<Control-c>', lambda event: cancel_ai())

    task_file_var = tk.StringVar(value="")
    cancel_flag = [False]

    print("Entering mainloop...")
    root.mainloop()
    print("Mainloop exited.")

if __name__ == "__main__":
    print("Script starting...")
    run_gui()
