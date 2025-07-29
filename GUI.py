# GUI class for creating adversarial examples using Whisper.
#
# libraries listed in the requirements.txt file + whisper_attack
#
# Author: Lukáš Hofman

import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import torchaudio

import whisper
import load_model

import pygame
import os
from screeninfo import get_monitors

from run_audio_attack import generate_example

"""
GUI class for creating adversarial examples using Whisper.
"""
class GUI:
    """
    Constructor for the GUI class.
    """
    def __init__(self):
        """
        Color scheme:
        Background: Dark Gray (#121212)
        Primary Color: Grayish blue (#607d8b)
        Accent: Magenta (#e91e63)
        Text: Light Gray (#e0e0e0)
        """
        self.path = os.path.dirname(os.path.abspath(__file__))
        
        self.root = tk.Tk()
        self.root.title("Adversarial Example Creation")
        
        width, height = self.get_current_monitor_resolution()

        self.screen_width = width - 12
        self.screen_height = height - 33
        # Change the appearance of the GUI
        self.root.geometry(f"{self.screen_width}x{self.screen_height}")
        # Set the universal background color
        self.root.option_add("*Background", "#121212")  # Dark Gray background
        self.root.option_add("*Foreground", "#e0e0e0")  # Light Gray text
        self.root.option_add("*Font", "Helvetica 12")   # Helvetica font size 12

        self.root.config(relief="flat")
        
        # Remove the window's border
        # self.root.overrideredirect(True)

        self.main_frame = tk.Frame(self.root)
        self.main_frame.option_add("*Background", "#1e1e1e")  # slightly lighter gray background
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.main_frame.columnconfigure(0, weight=1, minsize=400)  # Allow the first column to expand
        self.main_frame.columnconfigure(0, weight=3)
        
        self.app_label = tk.Label(self.main_frame, text="Adversarial Example Creation", font=("Helvetica", 24))
        self.app_label.pack(pady=20)
        
        self.parameters_frame = tk.Frame(self.main_frame,     
                                         bg="#1e1e1e",       # slightly lighter gray
                                         highlightthickness=1,
                                         highlightbackground="#444444",  # border color
                                         width=self.screen_width // 4,
                                         )
        self.parameters_frame.pack(pady=1, side="left", fill="y", padx=10)
        self.parameters_frame.pack_propagate(False)  # Prevent the frame from resizing to fit its contents

        # Create a label for the parameters frame
        self.parameters_label = tk.Label(self.parameters_frame, text="Parameters for Adversarial Attack", font=("Helvetica", 16))
        self.parameters_label.pack(pady=3, padx=20)

        height = max(self.screen_height // 8, 150)
        self.targeted_frame = tk.Frame(self.parameters_frame, height=height)
        self.targeted_frame.pack(pady=1, fill=tk.X)
        self.targeted_frame.pack_propagate(False)

        self.attack_type_frame = tk.Frame(self.parameters_frame, height=height, width=self.screen_width // 4)
        self.attack_type_frame.pack(pady=1, fill=tk.X)
        self.attack_type_frame.pack_propagate(False)
        
        self.attack_parameters_frame = tk.Frame(self.parameters_frame, width=self.screen_width // 4)
        self.attack_parameters_frame.pack(pady=1, fill=tk.BOTH, expand=True)
        self.attack_parameters_frame.pack_propagate(False)
                
        # Create frames for the different sections of the GUI
        height_original = max(self.screen_height // 5, 216)
        self.original_frame = tk.Frame(self.main_frame, height=height_original)
        self.original_frame.pack(pady=1, side="top", fill=tk.X)
        self.original_frame.pack_propagate(False)

        height_model_type = max(self.screen_height // 6, 200)
        self.model_type = tk.Frame(self.main_frame, height=height_model_type)
        self.model_type.pack(fill=tk.X)
        self.model_type.pack_propagate(False)
        
        height_generate_button = max(self.screen_height // 11, 95)
        self.generate_button_frame = tk.Frame(self.main_frame, height=height_generate_button)
        self.generate_button_frame.pack(pady=1, fill=tk.X)
        self.generate_button_frame.pack_propagate(False)

        self.generated_frame = tk.Frame(self.main_frame)
        self.generated_frame.pack(pady=1, fill=tk.BOTH, expand=True)
        
        self.save_frame = tk.Frame(self.generated_frame)
        self.save_frame.pack(fill=tk.BOTH, expand=True)
        
        self.exit_frame = tk.Frame(self.main_frame, height=max(self.screen_height // 10, 108))
        self.exit_frame.pack(side = tk.BOTTOM, pady=1, fill=tk.BOTH)
        self.exit_frame.pack_propagate(False)

        # Initialize Pygame mixer for audio playback
        pygame.mixer.init()

        # Set the icon using a .png or .gif file
        self.icon = tk.PhotoImage(file=os.path.expanduser(self.path + "/pictures/32_32_icon.png"))
        self.root.iconphoto(True, self.icon)


        # Validation commands
        def validate_int(value):
            if value == "" or value.isdigit():
                return True
            return False
        def validate_float(value):
            try:
                if value == "" or float(value) >= 0:
                    return True
                return False
            except ValueError:
                return False
        self.vint = (self.root.register(validate_int), '%P')  # Validate integer input
        self.vfloat = (self.root.register(validate_float), '%P')
        # Variables
        self.model = load_model.load_whisper_model("base")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = 16000 # whisper audio sample rate
        
        # Variables to hold audio and filepath
        self.original_audio_filepath = os.path.expanduser(self.path + "/data/sample-000001.wav")  # Default audio file path
        self.audio = None
        self.adversarial_audio = None

        self.epsilon = tk.DoubleVar(value=0.01)  # Default epsilon value
        self.sigma = tk.DoubleVar(value=0.01)
        self.max_iter = tk.IntVar(value=200)
        self.snr = tk.IntVar(value=35)  # Signal-to-noise ratio
        self.learning_rate = tk.DoubleVar(value=0.01)
        self.population_size = tk.IntVar(value=50)  # Population size for genetic attack
        self.max_decr = tk.IntVar(value=8)  # Maximum number of decrease attempts for CW attack
        self.const = tk.DoubleVar(value=4.00)  # Trade-off constant for CW attack
        self.decrease_factor_eps = tk.DoubleVar(value=0.8)  # Decrease factor for epsilon in CW attack
        
        self.width = 18
        self.height = 1
        
        # Load Audio button and label
        self.load_audio_button = tk.Button(self.original_frame, text="Load Audio", command=self.load_audio, 
                                           background="#607d8b", font=("Helvetica", 20),
                                           width=self.width,
                                           height=self.height,)
        self.load_audio_button.pack(pady=20, padx=20)

        self.audio_label = tk.Label(self.original_frame, text="No audio loaded", font=("Helvetica", 14))
        self.audio_label.pack(pady=10)

        self.play_original_button = tk.Button(self.original_frame, text="Play original audio", command=self.play_original_audio, state=tk.NORMAL, background="#607d8b", font=("Helvetica", 14))        
        self.stop_original_button = tk.Button(self.original_frame, text="Stop original audio", command=self.stop_audio, state=tk.NORMAL, background="#607d8b", font=("Helvetica", 14))


        # Model Type selection
        model_type_options = ['tiny', 'base', 'small', 'medium', 'large']
        self.model_type_var = tk.StringVar()
        self.model_type_var.set(model_type_options[1])

        self.model_type_label = tk.Label(self.model_type, text="Select Model Size:", font=("Helvetica", 16))
        self.model_type_label.pack(pady=15)        

        self.model_type_menu = tk.OptionMenu(self.model_type, self.model_type_var, command=self.update_model, *model_type_options)
        self.model_type_menu.config(background="#607d8b",
                               width=self.width,
                               height=self.height,
                               font=("Helvetica", 24))
        self.model_type_menu.pack(pady=15)

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = tk.Label(self.model_type, textvariable=self.status_var, font=("Helvetica", 16))

        # Targeted Attack Checkbox
        self.targeted_var = tk.BooleanVar()
        self.targeted_checkbox = tk.Checkbutton(self.targeted_frame, text="Targeted Attack", variable=self.targeted_var, 
                                                  command=self.update_attack_options, 
                                                  background="#1e1e1e", 
                                                  foreground="#e0e0e0",
                                                  font=("Helvetica", 16)
                                                  )
        self.targeted_checkbox.pack(pady=10)

        # Target Text Entry (Only visible when Targeted Attack is selected)
        self.target_text_label = tk.Label(self.targeted_frame, text="Target Text:", font=("Helvetica", 16))
        self.target_text_entry = tk.Entry(self.targeted_frame, 
                                          width=28,
                                          font=("Helvetica", 15))

        # Attack Type selection
        self.attack_type_var = tk.StringVar()
        self.attack_type_var.trace_add("write", self.on_attack_type_change)
        self.attack_type_label = tk.Label(self.attack_type_frame, text="Type of attack:")
        self.attack_type_label.pack(pady=20)

        self.attack_type_dropdown = tk.OptionMenu(self.attack_type_frame, 
                                                  self.attack_type_var, 
                                                  value="")
        self.attack_type_dropdown.config(background="#607d8b")
        self.attack_type_dropdown.pack(pady=20)

        # Epsilon Entry
        self.epsilon_label = tk.Label(self.attack_parameters_frame, text="Enter Epsilon:") # Or learning rate
        self.epsilon_entry = tk.Entry(self.attack_parameters_frame, textvariable=self.epsilon, validate='key', validatecommand=self.vfloat)

        # Sigma Entry
        self.sigma_label = tk.Label(self.attack_parameters_frame, text="Enter Sigma:")
        self.sigma_entry = tk.Entry(self.attack_parameters_frame, textvariable=self.sigma, validate='key', validatecommand=self.vfloat)
        
        # Number of Iterations Entry
        self.max_iter_label = tk.Label(self.attack_parameters_frame, text="Number of iterations:")
        self.max_iter_entry = tk.Entry(self.attack_parameters_frame, textvariable=self.max_iter, validate='key', validatecommand=self.vint)
        
        # SNR Entry
        self.snr_label = tk.Label(self.attack_parameters_frame, text="Signal-to-Noise Ratio (SNR):")
        self.snr_entry = tk.Entry(self.attack_parameters_frame, textvariable=self.snr, validate='key', validatecommand=self.vint)
        
        # Learning Rate Entry
        self.learning_rate_label = tk.Label(self.attack_parameters_frame, text="Learning Rate:")
        self.learning_rate_entry = tk.Entry(self.attack_parameters_frame, textvariable=self.learning_rate, validate='key', validatecommand=self.vfloat)
        
        # Population Size Entry (for Genetic Attack)
        self.population_size_label = tk.Label(self.attack_parameters_frame, text="Population Size:")
        self.population_size_entry = tk.Entry(self.attack_parameters_frame, textvariable=self.population_size, validate='key', validatecommand=self.vint)
        
        # Max Decrease Attempts Entry (for CW Attack)
        self.max_decr_label = tk.Label(self.attack_parameters_frame, text="Max Decrease Attempts:")
        self.max_decr_entry = tk.Entry(self.attack_parameters_frame, textvariable=self.max_decr, validate='key', validatecommand=self.vint)
        
        # Constant Entry (for CW Attack)
        self.const_label = tk.Label(self.attack_parameters_frame, text="Trade-off constant (c):")
        self.const_entry = tk.Entry(self.attack_parameters_frame, textvariable=self.const, validate='key', validatecommand=self.vfloat)
        
        # Decrease Factor Epsilon Entry (for CW Attack)
        self.decrease_factor_eps_label = tk.Label(self.attack_parameters_frame, text="Decrease Factor Epsilon:")
        self.decrease_factor_eps_entry = tk.Entry(self.attack_parameters_frame, textvariable=self.decrease_factor_eps, validate='key', validatecommand=self.vfloat)
        
        universal_attack_label_width = self.screen_width // 4 - 10
        # Label giving information about the universal attack
        self.universal_attack_label = tk.Label(self.attack_parameters_frame, 
                                               text="Using a pretrained universal attack with Epsilon = 0.02. \
                                               If you wish to use a different perturbation, you have to train it using the shell scripts provided in the repository. (See README.md)\
                                               Due to hardware limitations, whisper-large uses perturbation of whisper-medium.",
                                               font=("Helvetica", 16),
                                               fg="#e91e63",
                                               wraplength=universal_attack_label_width,
                                               justify="left",
                                           )

        # Update the options in the attack type dropdown based on the checkbox
        self.targeted_var.trace_add('write', self.update_attack_options)
        self.update_attack_options()

        # Create Adversarial Attack button
        self.create_attack_button = tk.Button(
            self.generate_button_frame, 
            text="Generate example", 
            command=self.create_adversarial_attack,
            background="#607d8b",
            font=("Helvetica", 20),
            width=self.width,
            height=self.height,
            state=tk.DISABLED
        )
        self.create_attack_button.pack(pady=25)

        # Label to display "Generating..." while the attack is being created
        self.original_transcription_label = tk.Label(self.generated_frame, text="Original transcription: ", font=("Helvetica", 14))
        self.example_transcription_label = tk.Label(self.generated_frame, text="Adversarial transcription: ", font=("Helvetica", 14))

        self.play_augmented_button = tk.Button(self.generated_frame, text="Play augmented audio", command=self.play_augmented_audio, 
                                               state=tk.NORMAL, font=("Helvetica", 14), background="#607d8b")
        self.stop_augmented_button = tk.Button(self.generated_frame, text="Stop augmented audio", command=self.stop_audio, 
                                               state=tk.NORMAL, font=("Helvetica", 14), background="#607d8b")

        self.save_button = tk.Button(self.save_frame, text="Save example", command=self.save_adversarial_audio, 
                                     background="#607d8b", font=("Helvetica", 20),
                                     width=self.width,
                                     height=self.height)

        self.exit_button = tk.Button(self.exit_frame, text="Exit", command=self.quit, 
                                     background="#607d8b", font=("Helvetica", 20),
                                     width=self.width,
                                     height=self.height)
        self.exit_button.pack(side=tk.BOTTOM, pady=30)
    
    """
    Function that returns the current monitor resolution.
    """
    def get_current_monitor_resolution(self):
        x = self.root.winfo_x()
        y = self.root.winfo_y()
        
        # Find which monitor the window is on (simplified)
        for m in get_monitors():
            if m.x <= x <= m.x + m.width and m.y <= y <= m.y + m.height:
                return (m.width, m.height)
        return (self.root.winfo_screenwidth(), self.root.winfo_screenheight())
    
    """
    Function to handle changes in the attack type dropdown.
    """
    def on_attack_type_change(self, *args):
        attack_type = self.attack_type_var.get()
        if attack_type == "SNR PGD":
            self.pack_forget()
            self.max_iter_label.pack(pady=10)
            self.max_iter_entry.pack(pady=10)
            self.snr_label.pack(pady=10)
            self.snr_entry.pack(pady=10)
        elif attack_type == "PGD":
            self.pack_forget()
            self.max_iter_label.pack(pady=10)
            self.max_iter_entry.pack(pady=10)
            self.epsilon_label.pack(pady=10)
            self.epsilon_entry.pack(pady=10)
        elif attack_type == "Modified CW" or attack_type == "CW":
            self.pack_forget()
            self.epsilon_label.pack(pady=10)
            self.epsilon_entry.pack(pady=10)
            self.max_iter_label.pack(pady=10)
            self.max_iter_entry.pack(pady=10)
            self.learning_rate_label.pack(pady=10)
            self.learning_rate_entry.pack(pady=10)
            self.max_decr_label.pack(pady=10)
            self.max_decr_entry.pack(pady=10)
            self.const_label.pack(pady=10)
            self.const_entry.pack(pady=10)
            self.decrease_factor_eps_label.pack(pady=10)
            self.decrease_factor_eps_entry.pack(pady=10)            
        elif attack_type == "Genetic":
            self.pack_forget()
            self.epsilon_label.pack(pady=10)
            self.epsilon_entry.pack(pady=10)
            self.max_iter_label.pack(pady=10)
            self.max_iter_entry.pack(pady=10)
            self.population_size_label.pack(pady=10)
            self.population_size_entry.pack(pady=10)
        elif attack_type == "Rand":
            self.pack_forget()
            self.sigma_label.pack(pady=10)
            self.sigma_entry.pack(pady=10)
        elif attack_type == "Universal PGD" or attack_type == "Universal PGD with noise":
            self.pack_forget()
            self.universal_attack_label.pack(pady=10)
        else:
            self.pack_forget()

    def pack_forget(self):
        """
        Function to hide the main frame.
        This is used when the application is closed.
        """
        self.epsilon_label.pack_forget()
        self.epsilon_entry.pack_forget()
        self.sigma_label.pack_forget()
        self.sigma_entry.pack_forget()
        self.max_iter_label.pack_forget()
        self.max_iter_entry.pack_forget()
        self.snr_label.pack_forget()
        self.snr_entry.pack_forget()
        self.learning_rate_label.pack_forget()
        self.learning_rate_entry.pack_forget()
        self.max_decr_label.pack_forget()
        self.max_decr_entry.pack_forget()
        self.const_label.pack_forget()
        self.const_entry.pack_forget()
        self.decrease_factor_eps_label.pack_forget()
        self.decrease_factor_eps_entry.pack_forget()        
        self.population_size_label.pack_forget()
        self.population_size_entry.pack_forget()
        self.universal_attack_label.pack_forget()

    """
    Function to quit the application.
    """
    def quit(self):
        tmp_dir = os.path.expanduser(self.path + "/tmp")
        if os.path.exists(tmp_dir):            
            os.remove(tmp_dir + "/example.wav")
            os.rmdir(tmp_dir)
        self.root.quit()
    
    """
    Function to load audio file from the file dialog.
    After a file is selected, the audio is loaded and the play and stop buttons are displayed.
    """
    def load_audio(self):
        initial_dir = os.path.expanduser(self.path + "/data")
        self.original_audio_filepath = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio files", "*.wav")],
            initialdir=initial_dir)
        
        self.audio = whisper.load_audio(self.original_audio_filepath, sr=self.sample_rate)
        self.create_attack_button.config(state=tk.NORMAL)  # Enable the create attack button
        if self.original_audio_filepath:
            pygame.mixer.music.load(self.original_audio_filepath)

            self.play_original_button.pack(side=tk.LEFT, padx=100, pady=20)
            self.stop_original_button.pack(side=tk.RIGHT, padx=100, pady=20)
            
            print(f"Loaded original audio file: {self.original_audio_filepath}")
            self.audio_label.config(text=f"Loaded Audio: {self.original_audio_filepath}")

    """
    Function that save the adversarial audio to a file.
    """
    def save_adversarial_audio(self):
        # Prompt user to choose file path for saving the adversarial audio
        filename = self.original_audio_filepath.split("/")[-1]
        default_filename = "adversarial_" + filename

        file_path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
            initialfile=default_filename,
            title="Save the adversarial audio",
            initialdir=os.path.dirname(self.original_audio_filepath)
        )
        
        if file_path:
            # Save the adversarial audio
            torchaudio.save(file_path, self.adversarial_audio, self.sample_rate)
            print(f"Adversarial audio saved to: {file_path}")
        else:
            print("Save operation cancelled.")

    """
    Starts playing the original audio by loading it into Pygame mixer and playing it.
    """
    def play_original_audio(self):
        if self.original_audio_filepath:
            pygame.mixer.music.load(self.original_audio_filepath)
            print("Playing original audio")
            pygame.mixer.music.play()

    """
    Function to play the augmented audio using Pygame mixer by saving the audio to a temporary file.
    The augmented audio must be saved to a temporary file before playing because otherwise it would not play in WSL2 as there is no audio output.
    """
    def play_augmented_audio(self):
        pygame.mixer.music.load(os.path.expanduser(self.path + "/tmp/example.wav"))
        print("Playing augmented audio")
        pygame.mixer.music.play()

    """
    Stops currently playing audio.
    """
    def stop_audio(self):
        pygame.mixer.music.stop()
        print("Original audio stopped")

    """
    Updates the model based on the selected model size.
    """
    def update_model(self, *args):
        model_size = self.model_type_var.get()
        self.status_var.set(f"Loading model '{model_size}'... Please wait.")
        self.status_label.pack(pady=10)
        
        # To avoid blocking the GUI
        self.root.after(10, self.load_model, model_size)

    def load_model(self, model_size):
        try:
            self.model = load_model.load_whisper_model(model_size)
            self.status_label.pack_forget()
        except Exception as e:
            self.status_var.set("Failed to load model.")
            messagebox.showerror("Error", f"Error loading model: {e}")

    """
    Updates the attack options based on the targeted checkbox.
    If targeted is selected, the attack type is set to targeted attacks and the target text entry is displayed.
    If targeted is not selected, the attack type is set to untargeted attacks.
    """
    def update_attack_options(self, *args):
        self.on_attack_type_change()
        if self.targeted_var.get():
            attack_type_options = ["Modified CW", "CW"]
            self.target_text_label.pack(pady=10)
            self.target_text_entry.pack(pady=10)
        else:
            attack_type_options = ["SNR PGD", "PGD", "Genetic", "Rand", "Universal PGD", "Universal PGD with noise"]
            self.target_text_label.pack_forget()
            self.target_text_entry.pack_forget()

        self.attack_type_var.set(attack_type_options[0])
        self.attack_type_dropdown['menu'].delete(0, 'end')
        for option in attack_type_options:
            self.attack_type_dropdown['menu'].add_command(label=option, command=tk._setit(self.attack_type_var, option))

    
    def _get_iters(self):
        nb_iter = self.max_iter.get()
        return f"nb_iter: {nb_iter}\n"
    def _setup_rand(self):
        sigma = self.sigma.get()
        return f"sigma: {sigma}\n"
    def _setup_genetic(self):
        overrides = self._setup_pgd()
        population_size = self.population_size.get()
        return overrides + f"population_size: {population_size}\n"
    def _setup_pgd(self):
        overrides = self._get_iters()
        eps = self.epsilon.get()
        return overrides + f"eps: {eps}\n"
    def _setup_snr_pgd(self):
        overrides = self._get_iters()
        snr = self.snr.get()
        return overrides + f"snr: {snr}\n"
    def _setup_cw(self):
        overrides = self._get_iters()
        eps = self.epsilon.get()
        # Limits the number of times the attack reduces the perturbation strength
        max_decr = self.max_decr.get()
        # Controls the balance between perturbation magnitude and misclassification confidence in the CW loss function.
        const = self.const.get()
        lr = self.learning_rate.get()
        decrease_factor_eps = self.decrease_factor_eps.get()
        target = self.target_text_entry.get()
        print(f"Target: {target}")
        return overrides + f"eps: {eps}\nmax_decr: {max_decr}\nconst: {const}\nlr: {lr}\ndecrease_factor_eps: {decrease_factor_eps}\ntarget_sentence: {target}\n"
    def _setup_modified_cw(self):
        return self._setup_cw()
    def _setup_universal(self):
        return ""

    def return_overrides_function(self, attack_type):
        """
        Returns the overrides function for the attack based on the selected options.
        """
        return {
            "SNR PGD": self._setup_snr_pgd,
            "PGD": self._setup_pgd,
            "Modified CW": self._setup_modified_cw,
            "CW": self._setup_cw,
            "Genetic": self._setup_genetic,
            "Rand": self._setup_rand,
            "Universal PGD": self._setup_universal,
            "Universal PGD with noise": self._setup_universal,
        }[self.attack_type_var.get()]
    
    """
    Function that creates the adversarial attack based on the selected options
    and displays the results.
    """
    def create_adversarial_attack(self):
        if self.audio is None:
            messagebox.showerror("Error", "No audio loaded.")
            return
        
        # Display the "Generating..." label
        self.forget_generated_frame()

        self.create_attack_button.config(text="Generating...", state=tk.DISABLED)
        self.root.update_idletasks()
        
        attack_type = self.attack_type_var.get().lower()
        print(f"Selected attack type: {attack_type}")
        attack_hparams = load_model.get_attack_hparams_file(attack_type)
        if attack_hparams is None:
            messagebox.showerror("Error", f"Attack type '{attack_type}' not found.")
            self.create_attack_button.config(text="Generate example", state=tk.ACTIVE)
            return
        overrides = f"root: {self.path}\n" + self.return_overrides_function(attack_type)()
                
        adv_audio = generate_example(
            attack_hparams,
            overrides,
            self.original_audio_filepath,
            self.audio,
            self.model
        )
        
        print(f"Attack Type: {attack_type}")

        # Temporarily save the adversarial audio to the same path as the original audio
        if not os.path.exists("tmp"):
            os.makedirs("tmp")
        torchaudio.save("tmp/example.wav", adv_audio, self.sample_rate)

        self.adversarial_audio = adv_audio

        # Print the results
        self.original_transcription = self.model.transcribe(self.original_audio_filepath)
        self.adversarial_transcription = self.model.transcribe("tmp/example.wav")
        print("Original transcription:", self.original_transcription)
        print("Adversarial transcription:", self.adversarial_transcription)

        self.prepare_generated_frame()

        self.create_attack_button.config(text="Generate example", state=tk.ACTIVE) # Remove the "Generating..." label
    
    """
    Function that prepares the generated frame with the adversarial example
    and displays the result aswell as the save button.
    """
    def prepare_generated_frame(self):
        self.original_transcription_label.config(text="Original transcription: " + self.original_transcription)
        self.original_transcription_label.pack(pady=10)
        self.example_transcription_label.config(text= "Adversarial transcription: " + self.adversarial_transcription)
        self.example_transcription_label.pack(pady=10)

        self.play_augmented_button.pack(side=tk.LEFT, padx=100, pady=20)
        self.stop_augmented_button.pack(side=tk.RIGHT, padx=100, pady=20)

        # Save Adversarial Audio
        self.save_button.pack(pady=10)

    """
    Function that hides previously generated frame.
    """
    def forget_generated_frame(self):
        self.original_transcription_label.pack_forget()
        self.example_transcription_label.pack_forget()
        self.play_augmented_button.pack_forget()
        self.stop_augmented_button.pack_forget()
        self.save_button.pack_forget()        

    """
    Function that runs the main loop of the GUI.
    """
    def mainloop(self):
        self.root.mainloop()

# Run the app
app = GUI()
app.mainloop()