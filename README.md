# Documentation for the Image-to-3D STL Model Generator Script

This script provides a utility to generate a 3D STL model from a 2D grayscale or color image. The resulting 3D model represents image intensity levels as varying depths. Below are detailed explanations of the script's components and usage:

---

## **Imports**
### Key Libraries:
1. **`cv2`**: OpenCV for image processing tasks (reading, resizing, preprocessing).
2. **`numpy`**: Array operations for vertex and face creation.
3. **`tkinter`**: GUI for visualization of image processing steps.
4. **`PIL`**: Image handling and display within the Tkinter GUI.
5. **`stl.mesh`**: For generating STL files.
6. **`tqdm`**: (Optional) Progress bars for iterative tasks.
7. **`asyncio`**: To handle asynchronous GUI updates and processing.

---

## **Functionality Overview**
### **1. Main Functionality**
- **Purpose**: Convert an image into a 3D model and save it as an STL file.
- **Process**:
  1. Load and preprocess the input image.
  2. Generate depth information based on pixel intensities.
  3. Visualize intermediate steps in a Tkinter GUI.
  4. Create the 3D mesh (vertices and faces).
  5. Save the mesh as an STL file.

### **2. Parameters**
- **`image_path`**: Path to the input image file.
- **`depth_levels`**: Number of layers representing the depth (default: `10`).
- **`target_width`**: Desired width of the final 3D model in millimeters (default: `100` mm).
- **`base_height`**: Height of the base plate in millimeters (default: `2` mm).
- **`max_dimension`**: Maximum dimension (pixels) for resizing large images (default: `1000`).

---

## **Detailed Function Descriptions**

### **1. `async def update_ui(root)`**
Handles periodic updates for the Tkinter GUI to keep it responsive during asynchronous processing.

```
async def update_ui(root):
    while True:
        try:
            root.update()
            await asyncio.sleep(0.01)
        except tk.TclError:
            break
```

- **Input**: 
  - `root`: The Tkinter root window.
- **Behavior**: Continuously updates the GUI unless the window is closed.

---

### **2. `async def generate_3d_model(image_path, depth_levels, target_width, base_height, max_dimension)`**
The primary function for generating the 3D model.

#### Workflow:
1. **Load Image**:
   - Reads the image using OpenCV and handles grayscale or color inputs.
   - Resizes the image if its dimensions exceed `max_dimension`.

2. **Preprocess Image**:
   - Applies smoothing (Gaussian blur) and morphological transformations to remove noise.
   - Converts the image to normalized depth values.

3. **Visualize Steps**:
   - Displays original, grayscale, and depth visualization images in a Tkinter GUI.

4. **Generate 3D Mesh**:
   - Computes vertices and faces for the top surface, bottom surface, and walls.
   - Creates the STL model using the `numpy-stl` library.

5. **Save STL**:
   - Saves the 3D mesh as an STL file named `<image_name>_3d.stl`.

```
async def generate_3d_model(image_path, depth_levels=10, target_width=100, base_height=2, max_dimension=1000):
    # Full implementation in the provided script.
```

- **Inputs**:
  - `image_path`: Path to the input image.
  - `depth_levels`: Number of depth layers (default: 10).
  - `target_width`: Desired model width in millimeters (default: 100).
  - `base_height`: Height of the base in millimeters (default: 2).
  - `max_dimension`: Maximum dimension for resizing images (default: 1000).

- **Output**:
  - Saves the STL file to the same directory as the input image.

---

## **Usage**
### **Command Line Execution**
To run the script, use the following command:

```
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python img2depth.py path/to/image.png
```

Options: `--width` (mm, default 100), `--base` (mm, default 2), `--relief` (mm, default 4),
`--colors` (palette size, default 4), `--max-dimension` (px, default 1000), `--out-dir`, `--no-preview`.

### **Outputs**
Written to `<image name>_3d/` next to the image:

| File | What it is |
|---|---|
| `relief.stl` | The whole image as a slab; darker pixels are raised. |
| `cutout.stl` | Only the subject, same relief, no slab. The subject comes from the alpha channel, or from the border colour when there is no transparency. |
| `color_original.ply` | The cut-out with the image's original colours (view in MeshLab, Blender or Quick Look). |
| `color_palette.3mf` | The cut-out split into `--colors` flat colours, one coloured part each, for multi-material printing. |
| `color_palette_parts/*.stl` | The same parts as separate STLs, named by colour. |

Run the tests with `.venv/bin/pytest`.

## **Examples** (Ultimaker Cura for 3D printing)
![image](https://github.com/user-attachments/assets/ff2bafa7-1fd0-491c-b700-424ed9d4fae6)
![image](https://github.com/user-attachments/assets/0688eeeb-ec5c-430f-9cfa-83e82b817070)
![image](https://github.com/user-attachments/assets/1a2f8d9c-f569-470f-b7ed-ce4b926793f9)
