
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
from joblib import load
from sklearn.preprocessing import MinMaxScaler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# Scaler initialization
min_values = np.array([6.0, 88.7, 140.0, 45.0, 40.0, 0.0, 0.0, 16.0, 8377892.333, 7984488.333])
max_values = np.array([30.0, 300.0, 426.4, 200.0, 165.0, 100.0, 100.0, 24, 262295552, 218764745.5])
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(np.array([min_values, max_values]))

# Model loading
def load_model(label_name):
    model_path = f"saved_models/{'XGB' if label_name in ['Ki', 'Mj,R'] else 'GB'}_best_model({label_name}).joblib"
    try:
        model = load(model_path)
    except FileNotFoundError:
        print(f"Model file {model_path} not found.")
        return None
    return model

models = {label: load_model(label) for label in ["Ki", "Mj,R", "Mmax", "Qu"]}

# Prediction function
def predict(features):
    features_array = np.array([features]).astype(np.float64)
    scaled_features = scaler.transform(features_array)
    results = {label: models[label].predict(scaled_features)[0] if models[label] else 'Model not loaded' for label in models}
    return results

# Global plot configuration
plot_colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
current_color_index = 0
figure = Figure(figsize=(6, 4), dpi=100)
ax = figure.add_subplot(111)
ax.set_title('Moment-rotation plot', fontsize=14)
ax.set_xlabel('Rotation (mrad)', fontsize=12)
ax.set_ylabel('Moment (kNm)', fontsize=12)
ax.grid(True)

def setup_canvas():
    global canvas, toolbar
    canvas = FigureCanvasTkAgg(figure, curve_frame)
    canvas.draw()
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)
    

    toolbar = NavigationToolbar2Tk(canvas, curve_frame)
    

    for tool in toolbar.winfo_children():
        if isinstance(tool, tk.Button) and tool['text'] == 'Save':
            tool['command'] = save_figure_with_options
    
    toolbar.update()

def save_figure_with_options():
    file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("JPEG files", "*.jpg")])
    if file_path:
        if file_path.endswith('.pdf'):
            figure.savefig(file_path, format='pdf')
        elif file_path.endswith('.jpg') or file_path.endswith('.jpeg'):
            figure.savefig(file_path, format='jpg')
        else:
            figure.savefig(file_path)

def plot_bilinear_curve(Ki, Mj_R, Mmax, Qu):
    global current_color_index, plot_colors
    A = (0, 0)
    B = (Mj_R / Ki, Mj_R) if Ki else (0, 0)
    C = (Qu, Mmax)
    color = plot_colors[current_color_index]
    ax.plot([A[0], B[0], C[0]], [A[1], B[1], C[1]], marker='o', color=color, label=f'Plot {current_color_index + 1}')
    ax.legend()
    
    canvas.draw()
    current_color_index = (current_color_index + 1) % len(plot_colors)

def clear_plot():
    global current_color_index
    if messagebox.askyesno("Clear Plot", "Do you want to remove all plots?"):
        ax.clear()
        ax.set_title('Moment-rotation plot', fontsize=14)
        ax.set_xlabel('Rotation (mrad)', fontsize=12)
        ax.set_ylabel('Moment (kNm)', fontsize=12)
        ax.grid(True)
        canvas.draw()
        current_color_index = 0

        # Clear inputs
        for entry in entries:
            entry.delete(0, tk.END)

        # Clear outputs
        for label in output_widgets:
            output_widgets[label].config(text='')
    else:
        # Clear only inputs
        for entry in entries:
            entry.delete(0, tk.END)

        # Clear only outputs
        for label in output_widgets:
            output_widgets[label].config(text='')

# UI Setup
root = tk.Tk()
root.title("")
root.configure(bg='#F0F0F0')
main_title = ttk.Label(root, text="Stainless-steel Flush End-plate Beam-to-column Connections Response ", font=('Times New Roman', 20, 'bold'))
main_title.grid(row=0, column=0, columnspan=2, pady=(20, 10))

feature_frame = ttk.LabelFrame(root, text="Feature Inputs", padding=20)
output_frame = ttk.LabelFrame(root, text="Model Outputs", padding=20)
curve_frame = ttk.Frame(root, borderwidth=2, relief="groove", padding=10)
info_frame = ttk.LabelFrame(root, text="Information", padding=20)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
feature_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
output_frame.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
curve_frame.grid(row=1, column=1, rowspan=2, padx=20, pady=10, sticky="nsew")
info_frame.grid(row=3, column=0, padx=20, pady=10, sticky="nsew")

# Information text area
info_text = tk.Text(info_frame, height=4, width=50, bg='#D3D3D3', fg='#333333', font=('Arial', 10), wrap="word")
info_content = """This GUI is developed by SINA SARFARAZI
Department of Science and Technology, University of Naples “Parthenope”, ITALY
Email: sina.srfz@gmail.com"""
info_text.insert(tk.END, info_content)
info_text.config(state=tk.DISABLED)
info_text.pack(expand=True, fill=tk.BOTH)

# Entry fields for input features
labels = ["End-plate thickness: tep(mm):", "End-plate width: bep(mm):", "End-plate height: hep(mm):",
          "Horizontal distance between bolts: gi(mm):","Spacing between bolts in tension and compression: Pi(mm):",
          "Spacing between the tension bolts: Pt(mm):", "Spacing between the compression bolts: Pc(mm):",
          "Bolt diameter: Db(mm):", "Column second moment of inertia: Ixxc(mm^4):",
          "Beam second moment of inertia: Ixxb(mm^4):"]
entries = [ttk.Entry(feature_frame, width=10) for _ in labels]
for i, (label, entry) in enumerate(zip(labels, entries)):
    ttk.Label(feature_frame, text=label).grid(row=i, column=0, padx=10, pady=5, sticky='w')
    entry.grid(row=i, column=1, padx=10, pady=5, sticky='w')

# Output labels
output_labels = {
    "Ki": "Initial rotational stiffness: Sj,ini(MNm/rad)",
    "Mj,R": "Plastic moment resistance: Mj,R(kN.m)",
    "Mmax": "Maximum moment resistance: Mj,max(kN.m)",
    "Qu": "Ultimate rotation: \u03A6j,u(mrad)"  # phi_u symbol
}

# Function to handle button click events
def submit():
    try:
        # Collect inputs
        feature_values = [float(entry.get()) for entry in entries]
        # Prediction
        results = predict(feature_values)
        # Update the output labels in the GUI
        for label, result_label in output_labels.items():
            result_value = results.get(label, 0)
            output_result = f"{result_value:.2f}"
            output_widgets[label].config(text=output_result)
        # Plotting
        plot_bilinear_curve(results["Ki"], results["Mj,R"], results["Mmax"], results["Qu"])
    except ValueError:
        messagebox.showerror("Input error", "Please check your inputs. All fields must be filled with numeric values.")

# Buttons for predictions and clearing plots
predict_button = ttk.Button(output_frame, text="Predict & Plot", command=submit)
predict_button.place(relx=0.9, rely=0.1, anchor=tk.CENTER)
clear_button = ttk.Button(output_frame, text="Clear", command=clear_plot)
clear_button.place(relx=0.9, rely=0.4, anchor=tk.CENTER)

# Output widgets
output_widgets = {}
for i, (label, output_label) in enumerate(output_labels.items()):
    ttk.Label(output_frame, text=output_label).grid(row=i, column=0, padx=10, pady=5, sticky='w')
    output_widgets[label] = ttk.Label(output_frame, text="", font=('Arial', 10))
    output_widgets[label].grid(row=i, column=1, padx=10, pady=5, sticky='w')

# Call to initialize the canvas and toolbar
setup_canvas()

# Main application loop
root.mainloop()


