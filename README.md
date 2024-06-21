# basler_acA2040-90um
Python device adaptor: Basler acA2040-90um USB 3.0 camera.
## Quick start:
- Download the Basler GUI (pylon Software, blaze Software), open the "pylon Viewer" and test the camera (the version used here was "pylon 7.4.0 Camera Software Suite Windows").
- Run 'basler_acA2040-90um.py' to test the camera (requires the 'pypylon' module).

![social_preview](https://github.com/amsikking/basler_acA2040-90um/blob/main/social_preview.png)

## Details
- Install 'pypylon' with pip "The official python wrapper for the Basler pylon Camera Software Suite" (https://github.com/basler/pypylon). Check out the 'samples' folder for example code on how to do various things.
- For documentation see https://docs.baslerweb.com/cameras, select the camera model and pick through the documentation and options that apply to that model. Small snippets of Python example code provided. It's also possible to download an offline version (seemingly no .pdf option but the web format is great!).
- **Note:** So far the GUI, the documentation and the camera have been very co-operative. This seems like a solid platform for quick and convenient development.
