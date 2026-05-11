from PIL import Image

# Open your PNG image
img = Image.open("HjBM.png")

# Save as .ico with multiple sizes
img.save("HjBM.ico", sizes=[(256,256)])